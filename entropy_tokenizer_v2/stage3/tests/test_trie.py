from literal_codec.codebook.trie import PrefixFreeTrie


def test_prefix_conflict_detection():
    trie = PrefixFreeTrie()
    assert trie.insert("ab")
    assert not trie.can_insert("a")
    assert not trie.can_insert("abc")
    assert trie.can_insert("ba")
