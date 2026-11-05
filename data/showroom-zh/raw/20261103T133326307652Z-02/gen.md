# 展厅 NLU — 中文生成说明 v1

生成 **500 条简体中文、普通话口语风格**的展厅用户表达。每条包含一个意图和
基于原文字符位置的槽位标注。直接用自然中文创作，不要逐句翻译英文模板。
本说明包含完整任务；直接向指定目录写入数据，无需浏览或搜索项目。

## 1. 输出目录与记录 ID

调用方提供 **output directory** 和 **record-ID prefix**。缺少时向调用方询问，
不要自行选择。生成十个文件，每个 50 条，命名为 `part-01.jsonl` 至
`part-10.jsonl`。数量与覆盖要求针对整批 500 条，不是每个文件。

`record_id` 从 `<supplied-prefix>-000001` 编到 `<supplied-prefix>-000500`。
跨文件连续编号，续写时保留已有 ID。不要覆盖调用方创建的 `manifest.json`
和说明快照。独立生成，不复制其他生成批次。

## 2. 业务词汇

规范 ID 保留英文；用户说出的名称、别名和周围句子用中文。以下展示名称和
表达是起点，可以提出清晰自然的别名，但不得增加功能或虚构能力。

| 功能 ID | 中文名称 / 示例表达 | 部位 | 主题 |
| --- | --- | --- | --- |
| `dlp_headlights` | DLP大灯、DLP车灯、大灯 | outside | family, entertainment |
| `cabin_cinema` | 座舱影院、车载影院、移动影院 | inside | family, entertainment |
| `massage_seats` | 按摩座椅、座椅按摩 | inside | entertainment |
| `privacy_window` | 隐私车窗、隐私玻璃 | outside | privacy |

这些是业务指定标签。按摩座椅的主题为 `entertainment`，不要新增 `comfort`。

| 槽位类型 | 规范值 | 示例表达 |
| --- | --- | --- |
| `flow_ref` | 上表四个功能 ID | 上表对应的功能名称 / 别名 |
| `part` | `inside` | 车内、内饰、座舱里 |
| `part` | `outside` | 车外、外饰 |
| `theme` | `family` | 家庭、全家、家用 |
| `theme` | `entertainment` | 娱乐 |
| `theme` | `privacy` | 隐私、私密 |

- `name`：用户明确自报的姓名或称呼，如李明、小王、欧阳晨、Alex。自由文本，
  不需要规范值。不要把用户对别人的称呼标成用户姓名。
- `choice`：对给定选项列表的数字选择，如第二个、第2个、选项三、2号。
  本批使用 1 到 10；数字与汉字均需覆盖。不标规范化的数字值。
- 新别名的预期含义可用可选 `slot_hints` 说明，由仲裁者按上下文判断。
- 不要把“投影大灯”“智能大灯”“调光玻璃”等能力相关名称自动等同于目录功能。
  不使用含义不清的“室内 / 室外”来代替车内部位。

### 只标注实际表达的内容

- `我想体验座舱影院` 只有 `flow_ref`，不能从目录推断并添加 part 或 theme。
- `我想看看车内的家庭功能` 有 part 和 theme，不据此添加某个功能 ID。
- `座舱影院`、`隐私玻璃` 是完整的 `flow_ref`；不要再给内部字符添加 part/theme。
- 没有目录功能满足的部位/主题组合，也可以是有效的演示请求。

## 3. 意图和允许的槽位

每条一个意图。**一个意图可以带多个不同类型的槽位，每个槽位类型至多一次。**

| 意图 | 含义 | 允许的槽位 | 示例 |
| --- | --- | --- | --- |
| `greet` | 打招呼，可紧接自我介绍 | 无，或一个 name | 你好；你好我叫小王 |
| `affirm` | 单纯同意或确认 | 无 | 对的；可以啊 |
| `deny` | 单纯否定或不同意 | 无 | 不是；不对 |
| `inform` | 告知姓名、偏好、值或列表选择 | name、flow_ref、part、theme、choice 中一个或多个，每种至多一次 | 我叫李明；第二个吧；我更喜欢车内适合家庭的功能 |
| `request_flow` | 请求看/试/启动演示，或询问可体验哪些功能 | 可选 flow_ref、part、theme，每种至多一次 | 我想体验按摩座椅；有什么可以体验的 |
| `request_stop` | 暂停当前演示 | 无 | 先暂停一下 |
| `request_resume` | 继续中断的演示 | 无 | 继续刚才的演示 |
| `request_quit` | 退出整个展厅体验 | 无 | 退出这次体验 |
| `UNK` | 超出本批支持范围，或无法理解 | 无 | 帮我订个餐厅；蓝色星期三搬走味道 |

### 标注决策

- `我更喜欢车内适合家庭的功能` → `inform`，part + theme。
- `给我看看车内适合家庭的功能` → `request_flow`，part + theme。
- 单独说 `座舱影院` → `inform`，flow_ref。
- `第二个吧`、`选第二个`、`给我看看第二个` → `inform`，choice。
  明确列表选择优先于“看看”等请求用词。
- `你好我叫小王` → `greet`，name；`小王你好` 不标 name。
- `不对` 是 deny，不代表暂停或退出；`对` 是 affirm，不代表继续演示。
- 除“打招呼后自我介绍”外，每条保持一个行为。不要生成
  `你好，给我看看大灯`、`对，就选第二个`、`不对，我叫小王` 等复合行为。
- 不生成同一字段的多个值（如 `家庭和娱乐都行`）、排除偏好（如 `不要娱乐的`）
  或依赖推断的偏好（如仅凭 `我有孩子` 就标 family）。多个不同字段可以共存。
- 避开没有上下文就无法区分 deny/stop/quit 的 `不用了`、`算了`、`结束吧`。
  用自然但明确的控制表达区分暂停当前演示与退出整个体验。
- UNK 混合其他任务、不支持的展厅操作（如买车、询价）和少量无法理解的文本。
  演示意图明确但功能未知，不自动算 UNK；这类未解析功能请求暂不生成。

## 4. 字符跨度

偏移量按原始 `text` 的 **Unicode 码点**计数，从 0 开始，start 包含、end 不包含：
`text[start:end]` 就是槽位原文。不是字节、UTF-16 单元、词序号或模型 token 序号。

- 保持自然中文，不为了标注给汉字之间插空格，也不进行分词。
- `DLP` 按 D、L、P 三个字符计数；原文空格、逗号和其他标点也占字符位置。
- 槽位按出现顺序排列，不重叠；标注后不要再改文本而不更新偏移。
- 标完整姓名、功能名、部位和主题表达。排除周围的请求动词、连接成分和标点。
  如 `我想体验[按摩座椅]`、`适合[家庭]的`、`[车内]的功能`。
- choice 包含数词/序数和选择量词，例如 `选[第二个]吧`、`就[2号]`、`[选项三]`。
  `吧`、`啊` 等句末语气词不在槽位内；`两个座椅` 的数量不是列表 choice。
- 保留名称内部必要的空格/标点。不要标注隐含目录属性，也没有 task 槽位。

## 5. JSONL 格式与示例

UTF-8 JSONL，每行一个完整 JSON 对象，不加 Markdown 围栏、注释或外层数组。
必需字段：`record_id`、`text`、`intent`、`slots`。每个 slot 仅含 type/start/end。
生成阶段不分配 group_id、不分 train/dev/test，也不生成 token 或 BIO 标签。

以下 ID 仅演示格式，实际使用调用方提供的前缀。不要照抄示例充数。

```jsonl
{"record_id":"example-000001","text":"你好","intent":"greet","slots":[]}
{"record_id":"example-000002","text":"你好，我叫小王","intent":"greet","slots":[{"type":"name","start":5,"end":7}]}
{"record_id":"example-000003","text":"对的","intent":"affirm","slots":[]}
{"record_id":"example-000004","text":"不是","intent":"deny","slots":[]}
{"record_id":"example-000005","text":"我叫李明","intent":"inform","slots":[{"type":"name","start":2,"end":4}]}
{"record_id":"example-000006","text":"第二个吧","intent":"inform","slots":[{"type":"choice","start":0,"end":3}]}
{"record_id":"example-000007","text":"就第2个","intent":"inform","slots":[{"type":"choice","start":1,"end":4}]}
{"record_id":"example-000008","text":"车内的吧","intent":"inform","slots":[{"type":"part","start":0,"end":2}]}
{"record_id":"example-000009","text":"我更喜欢车内适合家庭的功能","intent":"inform","slots":[{"type":"part","start":4,"end":6},{"type":"theme","start":8,"end":10}]}
{"record_id":"example-000010","text":"座舱影院","intent":"inform","slots":[{"type":"flow_ref","start":0,"end":4}]}
{"record_id":"example-000011","text":"有什么可以体验的","intent":"request_flow","slots":[]}
{"record_id":"example-000012","text":"我想体验按摩座椅","intent":"request_flow","slots":[{"type":"flow_ref","start":4,"end":8}]}
{"record_id":"example-000013","text":"给我看看DLP大灯吧","intent":"request_flow","slots":[{"type":"flow_ref","start":4,"end":9}]}
{"record_id":"example-000014","text":"车内的娱乐功能有什么能体验的","intent":"request_flow","slots":[{"type":"part","start":0,"end":2},{"type":"theme","start":3,"end":5}]}
{"record_id":"example-000015","text":"先暂停一下","intent":"request_stop","slots":[]}
{"record_id":"example-000016","text":"继续刚才的演示","intent":"request_resume","slots":[]}
{"record_id":"example-000017","text":"退出这次体验","intent":"request_quit","slots":[]}
{"record_id":"example-000018","text":"帮我订个餐厅","intent":"UNK","slots":[]}
```

新表达可附带 `slot_hints`，slot 是该条记录 slots 数组的零基下标，value 是
预期规范值。这只是给仲裁者的提示，只用于 flow_ref、part、theme，不用于姓名或选择：

```json
{"record_id":"example-000019","text":"我想试试车里的电影院","intent":"request_flow","slots":[{"type":"flow_ref","start":4,"end":10}],"slot_hints":[{"slot":0,"value":"cabin_cinema"}]}
```

## 6. 整批覆盖要求

| 意图 | 数量 |
| --- | ---: |
| `request_flow` | 140 |
| `inform` | 140 |
| `greet` | 40 |
| `affirm` | 30 |
| `deny` | 30 |
| `request_stop` | 30 |
| `request_resume` | 30 |
| `request_quit` | 30 |
| `UNK` | 30 |
| **合计** | **500** |

- Inform：覆盖五种槽位，混合单槽和多槽表达；多槽包括 part + theme 等自然组合。
  功能覆盖四项，部位和主题覆盖所有规范值，不用同一模板批量替换来充数。
- Greet：20 条纯问候，20 条问候后自我介绍。
- Request flow：20 条无槽位、40 条含功能名、25 条仅部位、25 条仅主题、
  30 条部位 + 主题。含功能名的 40 条中约 10 条也明确表达部位或主题，不重复计数。
- Choice：覆盖 1–10、汉字/阿拉伯数字、短回答/完整选择句。
- 姓名：以自然中文姓名和称呼为主，包含少量拉丁字母姓名，不固定使用同一个人名。

## 7. 口语风格与自查

混合简短回答、完整句子、直接请求、礼貌请求和自然疑问。适量使用“吧、啊、嗯”
等，不要每句都加语气词。混合有标点与少标点的表达，不强行编造语音识别错误。
保留自然混合文本如 DLP、Alex 和数字。以短句为主，长度适合一轮展厅对话。

不要主要靠换标点、加“请”或替换功能名来制造差异。检查每条意图和槽位含义、
精确字符跨度、每种槽位至多一次、整批配额、ID 唯一性和重复文本。
只写生成数据；仲裁和分组由后续仲裁者完成。
