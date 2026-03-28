from spy import Transformer, SPECIAL_TOKENS
import tiktoken

# 注意：这里用的是标准三引号字符串，避免奇怪的转义问题
code = """f,ax=plt.subplots(2,2,figsize=(15,12))

# 1、模型
rf=RandomForestClassifier(n_estimators=500,random_state=0)
# 2、训练
rf.fit(X,Y)
# 3、重要性排序
pd.Series(rf.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,0])
# 4、添加标题
ax[0,0].set_title('Feature Importance in Random Forests')

ada=AdaBoostClassifier(n_estimators=200,learning_rate=0.05,random_state=0)
ada.fit(X,Y)
pd.Series(ada.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,1],color='#9dff11')
ax[0,1].set_title('Feature Importance in AdaBoost')

gbc=GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,random_state=0)
gbc.fit(X,Y)
pd.Series(gbc.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,0],cmap='RdYlGn_r')
ax[1,0].set_title('Feature Importance in Gradient Boosting')

xgbc=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
xgbc.fit(X,Y)
pd.Series(xgbc.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,1],color='#FD0F00')
ax[1,1].set_title('Feature Importance in XgBoost')

plt.show()"""

# 1. 把 Python 代码转换成 SimPy 表示
transformer = Transformer()
try:
    spy_code = transformer.parse(code)
except Exception as e:
    # 如果 tree_sitter / 语法解析出错，这里会打印错误并直接退出，避免 traceback
    print("解析 Python 到 SimPy 失败：", repr(e))
    print("请先确认 spy/build 下的 *.so 已正确编译，以及代码片段本身是合法 Python。")
    raise SystemExit(1)

print("=== SimPy 代码 ===")
print(spy_code)

# 2. 构造带有 SimPy SPECIAL_TOKENS 的 GPT‑4 tokenizer（和论文里的 token_count 一样）
base_enc = tiktoken.encoding_for_model("gpt-4")
enc = tiktoken.Encoding(
    name="gpt4-spy",
    pat_str=base_enc._pat_str,
    mergeable_ranks=base_enc._mergeable_ranks,
    special_tokens={
        **base_enc._special_tokens,
        **{v: i + 100264 for i, v in enumerate(SPECIAL_TOKENS)},
    },
)

# 3. 对 SimPy 代码编码并统计 token
ids = enc.encode(spy_code, allowed_special=set(enc._special_tokens.keys()))
print("\n=== 每个 token ===")
for i, tid in enumerate(ids, start=1):
    piece = enc.decode([tid])
    print(f"[{i}] id={tid:5d} piece={piece!r}")

print("\n=== Token 数量 ===")
print(len(ids))