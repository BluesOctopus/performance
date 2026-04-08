import re
from typing import List, Dict, Optional

class CompressionPromptBuilder:
    """
    Generates a 'decompression manual' prompt for LLMs to understand 
    the Entropy Tokenizer v2 compressed format.
    """

    def __init__(self, selected_skeletons: List[Dict] = None, replacement_map: Dict[str, str] = None):
        self.selected_skeletons = selected_skeletons or []
        self.replacement_map = replacement_map or {}

    def build_system_prompt(self, current_text: str = None) -> str:
        """
        Builds the base system prompt. If current_text is provided, 
        it filters the dictionary to only include used markers (Dynamic Dictionary).
        """
        prompt = [
            "You are an expert Python developer and code decompressor.",
            "The following code is compressed using the 'Entropy Tokenizer v2' protocol.",
            "Please follow these rules to decompress and understand the code:",
            "",
            "### Rule 1: Syntax Operators (<SYN_N>)",
            "High-frequency code structures (skeletons) are replaced by <SYN_N> markers followed by slot values.",
            "The mapping is as follows:"
        ]

        # Dynamic Filtering for Skeletons
        used_syns = set(re.findall(r'<SYN_(\d+)>', current_text)) if current_text else None

        for i, sk_dict in enumerate(self.selected_skeletons):
            if used_syns is not None and str(i) not in used_syns:
                continue
            sk = sk_dict['skeleton']
            # Replace {0}, {1} with placeholders for clarity in prompt
            display_sk = sk
            for j in range(sk_dict.get('num_slots', 0)):
                display_sk = display_sk.replace(f"{{{j}}}", f"[SLOT_{j}]")
            prompt.append(f"- <SYN_{i}>: `{display_sk}`")

        prompt.extend([
            "",
            "### Rule 2: Minimalist Indentation (<I>, <D>)",
            "- Indentation is largely implicit: a line ending with a colon (:) implies the next line is indented.",
            "- <I>: Explicit Indent (move 4 spaces right).",
            "- <D>: Explicit Dedent (move 4 spaces left).",
            "- Multiple <D><D> means multiple dedent levels.",
            "",
            "### Rule 3: Token Replacements",
            "- <VAR_N>, <STR_N>, <NUM_N>, etc., are placeholders for frequent global tokens.",
            "- <V_N> refers to local identifiers defined in the local registry."
        ])

        if self.replacement_map:
            prompt.append("\n### Global Token Registry:")
            # Dynamic Filtering for Global Tokens
            used_vars = set(re.findall(r'<(VAR|STR|NUM|ATTR|FSTR)_(\d+)>', current_text)) if current_text else None

            categories = {}
            for token, marker in self.replacement_map.items():
                m_match = re.search(r'<(VAR|STR|NUM|ATTR|FSTR)_(\d+)>', marker)
                if used_vars is not None and m_match:
                    if (m_match.group(1), m_match.group(2)) not in used_vars:
                        continue

                cat = re.sub(r'_\d+', '', marker.strip('<>'))
                if cat not in categories: categories[cat] = []
                categories[cat].append(f"{marker} -> `{token}`")
            
            for cat, items in categories.items():
                prompt.append(f"#### {cat}:")
                prompt.extend([f"  - {item}" for item in items]) # No more limit, it's already filtered

        return "\n".join(prompt)

    def build_file_context(self, compressed_text: str) -> str:
        """
        Extracts local registry from the compressed text and formats it as context.
        """
        registry_match = re.search(r'<R>(.*?)</R>', compressed_text)
        if not registry_match:
            return "No local registry found for this file."

        words = registry_match.group(1).split(',')
        context = ["### Local Identifier Registry for this file:"]
        for i, word in enumerate(words):
            context.append(f"- <V{i}>: `{word}`")
        
        return "\n".join(context)

    def generate_full_prompt(self, compressed_text: str) -> str:
        """
        Combines system prompt, file context, and the compressed code.
        """
        system = self.build_system_prompt(compressed_text)
        file_ctx = self.build_file_context(compressed_text)
        
        # Remove the registry header from the code display to avoid redundancy
        code_body = re.sub(r'<R>.*?</R>\n?', '', compressed_text)

        full_prompt = [
            "--- SYSTEM INSTRUCTIONS ---",
            system,
            "",
            "--- FILE CONTEXT ---",
            file_ctx,
            "",
            "--- COMPRESSED CODE ---",
            "```python",
            code_body,
            "```",
            "",
            "Please provide the fully decompressed Python code."
        ]
        return "\n".join(full_prompt)

if __name__ == "__main__":
    # Example usage
    example_skeletons = [
        {"skeleton": "def {0}({1}):", "num_slots": 2},
        {"skeleton": "if {0}:", "num_slots": 1}
    ]
    example_rmap = {"self": "<VAR_0>", "None": "<VAR_1>"}
    
    builder = CompressionPromptBuilder(example_skeletons, example_rmap)
    
    test_code = "<R>my_func,x,y</R>\n<SYN_0> <V0> <V1> <V2>\n    <SYN_1> <V1> == <VAR_1>\n        return <V2>"
    
    print(builder.generate_full_prompt(test_code))
