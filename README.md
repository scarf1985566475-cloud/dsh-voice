# dsh-voice — 录音转写 · 英译中同声传译 · 会议记录与纪要 · Obsidian 导出

给 DeepSeek Harness 加一套耳朵、记性和笔记本：

| 能力 | 说明 |
|---|---|
| **录音转写** | 任意 ffmpeg 可读的音频/视频 → 英文文字（含时间轴），可顺带翻译 |
| **同声传译** | 麦克风实时英文字幕 + 中文译文，在 DSH Web GUI 里直接看 |
| **会议记录** | 每次同传会话自动存档：逐句中英文 + 时间轴 + 元数据，落盘可回溯 |
| **会议纪要** | 一键把记录整理成中文纪要：概览 / 关键要点 / 决定事项 / 待办（含负责人）/ 风险 |
| **导出 Obsidian** | 纪要 + 全文写成 Obsidian 原生笔记（frontmatter、双链可用、待办是真正的 checkbox） |

全部识别在本机完成（MLX Whisper，Apple Silicon 加速）；只有英文文本会发给 DeepSeek 做翻译和总结。
音频**从不上传**。

---

## 一分钟上手

```bash
# 1. 环境（已在本项目创建好）
.venv-audio/bin/dsh-voice doctor          # 自检：ffmpeg / 模型 / 翻译 Key / 会议记录

# 2. 起服务（面板与 MCP 工具都依赖它）
.venv-audio/bin/dsh-voice serve
#    打开 http://127.0.0.1:8768/ 可以直接测试麦克风

# 3. 不开麦克风也能验证整条链路
.venv-audio/bin/dsh-voice selftest --live
```

在 DSH Web GUI 中：会话上方切到 **「同传」** 标签页，里面有两个模式：

- **实时** — 点「开始收听」，边听边出字幕；同时自动记录本次会议（可当场命名）。
- **会议记录** — 左侧是历史会议列表，选中后可看全文或纪要，点「生成纪要」出中文会议纪要。

---

## 架构

```
                    ┌──────────────────────────────────────────────┐
   麦克风 ──► 浏览器 │ DSH Web GUI「同传」面板（客户端插件）          │
                    │  实时：getUserMedia → 16kHz float32 → WS      │
                    │  会议记录：列表 / 全文 / 纪要 / 生成纪要        │
                    └───────────────┬──────────────────────────────┘
                                    │ ws://127.0.0.1:8768/live  (+ /meetings)
                    ┌───────────────▼──────────────────────────────┐
                    │ dsh-voice 本地服务（Python / FastAPI）         │
                    │  VAD 分段 → 滚动窗口 → MLX Whisper 识别        │
                    │  → 稳定句子提交 → DeepSeek 英译中              │
                    │  → 追加写入会议日志（JSONL，崩溃安全）          │
                    │  → 纪要：结构化 JSON → 本地渲染 Markdown       │
                    └───────────────┬──────────────────────────────┘
                                    │ 落盘
                    ┌───────────────▼──────────────────────────────┐
                    │ meetings/<id>/                               │
                    │   meta.json          标题/时长/段数/纪要状态   │
                    │   transcript.jsonl   逐句追加日志（不重写）     │
                    │   summary.md / .json 会议纪要                 │
                    └──────────────────────────────────────────────┘
                                    ▲
                    ┌───────────────┴──────────────────────────────┐
                    │ MCP stdio（11 个工具）                        │
                    │  转写/翻译/录音/设备/状态 + 会议列表/读取/      │
                    │  生成纪要/处理录音文件/改名                    │
                    └──────────────────────────────────────────────┘
```

插件包 `plugin/` 同时提供浏览器半边（面板 UI）与 Node 半边（守护本地服务进程）。

---

## 「同声传译」为什么是"同声"而不是"等说完再译"

关键在**提交时机**。只靠静音检测不够：一个人连续讲 60 秒不停顿，就会 60 秒什么都看不到。
所以会话持续重解码一小段滚动窗口，凡是"句尾已经落在音频前沿之后 1 秒以上"的句子（重新解码也不会变），
就立刻提交并翻译。讲者还在说话时字幕已经在滚动，这才是同传。

事件协议（WebSocket `/live`）：

| 方向 | 消息 | 说明 |
|---|---|---|
| 客户端 → 服务 | 二进制帧 | float32 小端、单声道、16kHz PCM |
| 客户端 → 服务 | `{"type":"config"}` / `{"type":"title"}` / `{"type":"flush"}` / `{"type":"ping"}` | 运行时开关 / 给本次会议命名 / 立即提交 / 心跳 |
| 服务 → 客户端 | `ready` | 采样率、生效配置、以及本次会议的 id 与标题 |
| 服务 → 客户端 | `partial` | 未提交窗口的预览，`id` 就是它将来提交时的 id |
| 服务 → 客户端 | `final` | 一句已提交的英文字幕（含时间与耗时） |
| 服务 → 客户端 | `translation` | 对应中文（`final:false` 为预览译文） |
| 服务 → 客户端 | `flushed` | 收到 `flush` 后**已提交并翻译完尾部**才回这个 ack（面板据此安全收尾） |
| 服务 → 客户端 | `error` | 非致命错误，会话继续 |

同 id 的 `partial` 会被 `final` **原地替换**，所以字幕不会重复堆积。

「停止」的握手：客户端发 `flush` → 服务把未提交窗口解码、提交、翻译完 → 回 `flushed` →
客户端才关连接。少了这一步，讲话最后一句的译文会因为连接关闭而丢失。

---

## 会议记录与纪要

### 记录是自动的

开始收听就创建一次会议记录；每次提交一句英文就**立刻落盘**，译文到达后再追加一条。
所以哪怕进程被 `kill -9`，已经听到的内容依然能读回来。

`transcript.jsonl` 是**只追加**的，一条一句，不重写整个文件：

```jsonl
{"kind":"segment","id":3,"start":8.3,"end":12.9,"en":"We decided to delay the migration.","asr_ms":430,"at":"..."}
{"kind":"translation","id":3,"zh":"我们决定推迟迁移。","ms":520}
```

读取时按 `id` 把译文折叠回句子；损坏的尾行、找不到句子的"孤儿译文"都会被跳过而不是让整份记录失败。

### 纪要是有结构的，不是一段散文

模型被要求输出 **JSON**（`overview / key_points / decisions / action_items / risks / topics`），
Markdown 由本机渲染。这样面板能把待办渲染成表格，模型答非所问时也只会退化成可读文本，而不是坏文档。

提示词里立了硬规矩：**只能依据记录内容，没说的写「未提及」**，人名/数字/日期/缩写原样保留。
`action_items` 每项都带 `task / owner / due`，未知的填「未提及」而不是猜。

长会议自动 map-reduce：先按句子边界切成 ≤12000 字符的块，逐块出要点笔记，再合并成最终纪要。

### 文件也有同样的待遇

已经录好的会议（录音、通话、视频音轨）走同一条链路，产出**结构完全一样**的记录：

```bash
.venv-audio/bin/dsh-voice ingest ~/Downloads/standup.m4a --title "每日站会"
```

文件路径的翻译是**批量**的（每 8 句一次请求）——文件没有延迟预算，一句一次调用既慢又贵。
实时路径仍然一句一次，因为那里延迟就是全部意义。

### HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/meetings?limit=N` | 会议列表（标题、时间、时长、段数、纪要状态） |
| GET | `/meetings/{id}` | 单个会议：元数据 + 逐句记录 + 纪要 |
| GET | `/meetings/{id}/transcript?bilingual=` | 纯文本全文 |
| POST | `/meetings/{id}/summary` | 生成纪要（`{"force":true}` 强制重算；已有则直接返回缓存） |
| POST | `/meetings/{id}/title` | 重命名会议 |
| DELETE | `/meetings/{id}` | 删除会议 |
| POST | `/recordings/summarize` | 把已有录音文件变成会议记录 + 纪要 |

---

## 本地模型：怎么选、为什么

```bash
dsh-voice benchmark                                   # 三个模型同场对比（约 1.5 分钟）
dsh-voice benchmark --text-file selftest/hard-lecture.txt   # 换成学术讲座稿再测一遍
DSH_VOICE_MODEL=small dsh-voice serve                 # 换模型：turbo / distil / small
```

`benchmark` 在**同一段音频**上量三件事：**切片延迟**（6 秒 = 一次实时提交，观众真正感觉到的那段延迟）、
**准确率**（对照已知讲稿算 WER，且分别测干净音频和 10 dB 信噪比的加噪音频）、**峰值内存**。
每个模型跑在独立子进程里，否则前一个模型的权重会算到后一个头上。

实测（Apple Silicon，同一份讲稿）：

| 模型 | 大小 | 峰值内存 | WER 干净 | WER 加噪 | 6s 切片 | 20s 切片 |
|---|---|---|---|---|---|---|
| `turbo`（默认） | 1614M | 2147M | 0.9% | **0.9%** | 2342ms | 3118ms |
| `distil`（仅英文） | 1509M | 2042M | 0.9% | 1.4% | 2009ms | 2367ms |
| `small` | 481M | 1231M | 1.4% | 1.9% | **572ms** | **1154ms** |

**结论：默认继续用 `turbo`。** 它抗噪最稳（0.9%），而 `distil` 只快 ~20%、抗噪差半档；
`small` 快 4 倍但准确率最差——**注意这是合成语音上的数字，真实讲座音频只会让差距更大**，
所以没有把它设成默认。三个模型都装好了，一条环境变量就能换：

- 想要**最短字幕延迟**、可以接受偶尔听错词 → `DSH_VOICE_MODEL=small`
- 只要**英文**、想要折中 → `DSH_VOICE_MODEL=distil`
- 中文/多语种或用自动识别 → 必须用 `turbo`（配错会给出**明确警告**，见 `dsh-voice doctor`）

### 顺手挖出来的一个大坑：把 temperature 钉死会关掉"救命重试"

原来代码里写了 `temperature=0.0` —— 这是很常见的写法，看起来只是"要确定性输出"。
但它同时**关掉了 Whisper 的兜底重试**：压缩比 / 对数概率那两道闸门一旦触发，
本应升温重解，钉死 0.0 就变成"错了也认"。实测同一段音频：

| 解码设置 | WER | 输出词数 |
|---|---|---|
| `temperature=0.0`（旧） | **72.6%** | 369（原文 215） |
| 保留升温阶梯（新） | 4.7% | 220 |
| 升温阶梯 + 不再喂回上文（新，两者都开） | **0.9%** | 215 |

72.6% 不是"识别得差"，是**复读机**——模型在某一窗卡进循环，把同一句话重复吐了几百个词。
而这个输入和正常输入**听起来完全一样**：两个数组最大差值只有 5e-5（16 位量化误差），
一个 0.9%、一个 72.6%。也就是说，**同一段音频重跑一次就可能翻车**。

现在两条路径都：保留升温阶梯 + `condition_on_previous_text=False`（不再把上一窗的输出喂回给下一窗，
这正是复读的起点）。并且基准测试新增了 `worst` 列——同一段音频分别喂浮点和 16 位量化版本，
取**较差**的那个，这样"靠运气跑对"的模型藏不住。

> 这个修复的价值超过了换模型本身：它把一个"偶尔整段崩掉"的隐患变成了 0.9%。
> 回归测试 `tests/asr_config_test.py` 会盯着这两条设置，被改回去就红。

---

## 热重载：改哪一层需要做什么

| 改了什么 | 要不要手动操作 | 机制 |
|---|---|---|
| `plugin/client.js`（面板 UI） | **什么都不用做** | client-hmr 每 500ms 轮询 bundle 文件，变了就发 SSE，页面里的模块**原地换掉** |
| `cordis.patch.yml`（插件行/配置） | **什么都不用做** | profile 专门挂了 watch-only HMR 来热加载这个文件 |
| `dsh_voice/*.py`（后端） | 要重启服务，或用 `--reload` | 服务进程把模块读在内存里 |
| `plugin/index.js`（Node 半边） | 要重启 dsh | 宿主进程里 ESM 模块被缓存 |

### 面板 UI：改完就生效，不用刷新

实测（只加了一行注释）：

```
改之前  rev=e182c4417d48
改之后  rev=91e9b0637bbf
SSE     data: {"type":"rebuilt","id":"dsh-voice-interpreter","rev":"91e9b0637bbf"}
```

浏览器那一半收到 `rebuilt` 会**删掉旧模块记录 → 摘掉它注入的 `<style>` → 重新 import 并重新 apply**，
所以面板在原地换新。**不用刷新页面，也不需要 `pnpm run dev:web`**——那条命令是给需要 TS→JS 构建的
插件用的；本插件是手写 bundle，文件本身就是要跑的东西。

> ⚠️ 唯一要注意的：热替换会**卸载并重新挂载面板**。如果你正在录音，这一场会掉线
> （服务端会正常存档并 finalize，数据不丢，但要重新点「开始收听」，且那会是新的一场会议）。

### 后端 Python：`--reload`

```bash
.venv-audio/bin/dsh-voice serve --reload
```

实测：`touch dsh_voice/config.py` 后 worker 自动重启（日志 `WatchFiles detected changes ... Reloading`，
`/health` 的 uptime 从 31.0s 归零到 10.9s）。

代价：重启会**掐断正在进行的实时会话**，所以默认关闭，是开发时手动开的选项。日常用普通 `dsh-voice serve`。

### 录音的生命周期：切页面不会断

面板挂在 `conversation.view` 上，这个槽位是**会话级**的——切到别的标签页、切到别的对话，
组件都会被卸载。早期版本把会话状态放在组件里，于是**一离开面板录音就断**，这正是一个会议记录器最不该做的事。

现在会话状态在**模块作用域**（`voiceSession`），归属插件的生命周期，不属于任何组件：

| 动作 | 结果 |
|---|---|
| 切标签页 / 切对话 / 切到别的界面 | 只是**断开视图**，录音继续，字幕继续累积 |
| 切回来 | 面板**接着显示进行中的会议**（标题、已记录段数、字幕都在） |
| 点「停止」 | 正常收尾：flush → 翻译完 → 归档 |
| 插件被卸载（热重载 / 禁用） | 立刻交还麦克风（不等 flush 握手），服务端把会议存档 |

> 副作用要说清楚：切走之后**界面上看不到"正在录音"的提示**（面板不在渲染了）。
> 对"边听讲座边切去别处干活"是有意的取舍；如果你想要一个全局指示灯，可以再挂一个 `shell.overlay` 槽位。

回归测试 `tests/plugin_session_test.mjs` 直接盯着这条：**在没有挂载任何组件的情况下**驱动整条链路，
再模拟"挂上视图 → 摘掉视图"，断言会话仍然活着、麦克风没有被释放。

---

## 导出到 Obsidian

```bash
dsh-voice export --vaults                    # 看看 Obsidian 认识哪些 vault
dsh-voice export 20260923-150632-013-e16a    # 导出（默认写进 80-会议记录/）
dsh-voice export <id> --folder "会议/2026"    # 换目录
dsh-voice export <id> --no-transcript        # 只要纪要，不要逐句记录
```

导出的是一份**给 Obsidian 看的**笔记，不是纯文本倾倒：

- **frontmatter**：`title / created / meeting_id / duration / segments / source / tags / topics`
  —— dataview 可以直接查「这个月所有带 #会议记录 的会议」
- **待办事项是真 checkbox**：`- [ ] 准备供应商对比 — @Sarah — 📅 下周五`
  —— tasks 插件能跨库汇总（Markdown 表格做不到这件事）
- **逐句记录收在折叠 callout 里**（`> [!quote]-`），打开是全文，不打开不占地方
- **文件名稳定**：`2026-09-23 标题.md`，重复导出是**更新同一篇**，不会堆副本

两条路径，自动选择：

1. **直接写 vault 文件**（默认）——Obsidian 开不开都能用，不依赖任何插件；
2. **Local REST API**（配了 `DSH_VOICE_OBSIDIAN_API_URL` + `OBSIDIAN_API_KEY` 时）——
   PUT 给插件，正在运行的 Obsidian 立刻刷新；失败会自动退回路径 1。

Vault 解析顺序：`--vault` → `DSH_VOICE_OBSIDIAN_VAULT` → Obsidian 注册表里最后打开的那个
（`~/Library/Application Support/obsidian/obsidian.json`）。

---

## MCP 工具

接入后模型可直接调用（工具名带 `mcp__voice__` 前缀）：

| 工具 | 用途 |
|---|---|
| `voice_status` | 模型 / ffmpeg / 翻译 Key / 服务 / 会议数是否正常 |
| `transcribe_audio` | 转写任意 ffmpeg 可读文件，可选同时翻译 |
| `translate_to_chinese` | 英文 → 中文（可带上下文保持术语一致） |
| `list_audio_devices` | 列出麦克风设备 |
| `record_audio` | 用内置 ffmpeg 录音，可顺带转写 |
| `ensure_live_server` | 探测本地服务，必要时拉起 |
| `list_meetings` | 会议列表（最新在前） |
| `read_meeting` | 读一次会议：元数据 + 逐句记录 + 纪要 |
| `summarize_meeting` | 生成中文纪要（结构化），结果落盘复用 |
| `summarize_recording` | 录音文件一条龙：转写 → 翻译 → 存档 → 纪要 |
| `rename_meeting` | 给会议起个有意义的名字 |
| `list_obsidian_vaults` | Obsidian 认得哪些 vault |
| `export_meeting_to_obsidian` | 把会议导出成 Obsidian 笔记（纪要 + 全文 + checkbox 待办） |

例如在对话里直接说「把上次会议整理成纪要」，模型会 `list_meetings` → `summarize_meeting`；
说「导出到 Obsidian」就会调 `export_meeting_to_obsidian`。

---

## 配置项

环境变量（`DSH_VOICE_` 前缀，或直接同名）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MODEL` | `models/whisper-large-v3-turbo` | 模型：`turbo` / `distil` / `small`，或任意路径 / repo |
| `MODEL` | `turbo`（多语种） | 认 `turbo` / `distil`(仅英文) / `small`，也可直接给路径或 HF repo |
| `LANGUAGE` | `en` | 源语言；`auto` 或空串 = 自动识别（会强制用多语种模型） |
| `PORT` / `HOST` | `8768` / `127.0.0.1` | 服务监听地址 |
| `TRANSLATE` | `true` | 是否翻译 |
| `TRANSLATE_PARTIALS` | `true` | 预览句是否也翻译 |
| `PARTIAL_INTERVAL` | `1.2` | 预览解码间隔（秒） |
| `END_SILENCE_MS` | `700` | 多长静音判为一句结束 |
| `MAX_SEGMENT` | `24` | 单段硬上限（秒） |
| `RECORD` | `true` | 实时会话是否写入会议记录 |
| `MEETINGS_DIR` | `meetings/` | 会议记录根目录 |
| `API_BASE` / `API_MODEL` | `https://api.deepseek.com` / `deepseek-chat` | 翻译与纪要后端 |
| `OBSIDIAN_VAULT` | 注册表里最后打开的 vault | Obsidian vault 目录 |
| `OBSIDIAN_FOLDER` | `80-会议记录` | vault 内的目标目录 |
| `OBSIDIAN_API_URL` / `OBSIDIAN_API_KEY` | 空 | 配了就改走 Local REST API 推送 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | 模型下载镜像（huggingface.co 不可达时用） |

翻译 Key 解析顺序：`DEEPSEEK_API_KEY` 环境变量 → `$DSH_HOME/.credentials.yaml`。
`DSH_HOME` 未设置时默认 `~/.dsh-home`：如果你的 DSH 是从源码仓库跑的，就指向那个仓库里的
`.dsh-home`（本机实测用的就是这种方式）。测试脚本也认这个变量，所以换机器不用改代码。

---

## 命令

```bash
dsh-voice serve [--port N] [--no-translate] [--reload]   # 实时服务（面板/MCP 的后端）
dsh-voice transcribe FILE [--translate|--text] # 文件转写
dsh-voice translate "text"                     # 英文 → 中文
dsh-voice record --seconds 15 --transcribe     # 麦克风录音
dsh-voice devices                              # 设备列表

dsh-voice meetings [--limit N] [--text]        # 会议列表
dsh-voice show ID [--summary|--english-only|--json]  # 看全文或纪要
dsh-voice summarize ID [--force]               # 生成纪要
dsh-voice ingest FILE [--title T] [--text]     # 录音文件 → 记录 + 纪要
dsh-voice export ID [--vault V] [--folder F]   # 导出到 Obsidian
dsh-voice export --vaults                      # 列出 Obsidian vault

dsh-voice doctor                               # 依赖自检（含模型/语种是否匹配）
dsh-voice benchmark [--text-file F]            # 本地模型同场对比
dsh-voice download-model                       # 下载当前模型（约 1.6GB）
dsh-voice selftest [--live]                    # 端到端自检（无需麦克风）
```

---

## 测试

```bash
bash tests/run_all.sh          # 离线检查（无模型、无网络，秒级，9 组）
bash tests/run_all.sh --full   # 全链路（模型 + DeepSeek API，约 3 分钟）

# 也可以单独跑：
.venv-audio/bin/python tests/meetings_test.py            # 记录存储：崩溃容错、排序、纪要渲染
.venv-audio/bin/python tests/obsidian_export_test.py     # 笔记结构、文件名安全、重复导出
.venv-audio/bin/python tests/asr_config_test.py          # 解码配方与模型选择（防复读回归）
.venv-audio/bin/python tests/benchmark_test.py           # 基准测试自身：WER 算法、加噪、推荐规则
node tests/plugin_render_test.mjs                        # 面板渲染 + 纪要渲染器（含 XSS 安全）
node tests/plugin_host_test.mjs                          # 守护逻辑：复用而不抢占
node tests/plugin_session_test.mjs                       # 录音生命周期：切页面不能断（本 bug 的回归测试）
.venv-audio/bin/python tests/live_stream_test.py         # 实时会话 + 记录一致性
.venv-audio/bin/python tests/server_ws_test.py           # 真实 WebSocket + 会议 HTTP 接口
.venv-audio/bin/python tests/recording_pipeline_test.py  # 录音 → 转写 → 纪要（含事实核查）
```

关键不变量（都是先红过才写下来的）：

- 实时路径：**任何一句都不得提交两次**，且**每个已提交句都必须有译文**。
- 记录路径：**存下来的句子必须和推送给用户的事件逐字一致**，段数、译文一个都不能少。
- 纪要路径：**说过的数字、负责人、截止时间必须出现在纪要里** —— 只对不上的措辞不算。
- 导出路径：**重复导出只能更新同一篇笔记**，文件名不能带 Obsidian 拒绝的字符，标题不能出现两次。
- 解码路径：**升温阶梯必须在**、**不得把上一窗输出喂回下一窗** —— 少任何一条，同一段音频就可能整段复读。
- 模型选择：**英文专用模型绝不能喂中文**，自动识别必须落到多语种模型。
- 面板：纪要 Markdown **永远不能变成 HTML 注入**（`<script>` 只会以文本出现）。

---

## 在 DSH 里怎么接的

写入（工作区外，已授权）：

1. `.dsh-home/profiles/web/node_modules/dsh-voice-interpreter` → 软链到本目录的 `plugin/`
   （等价于 `dsh plugin --profile web add <path>`，只是不动 lockfile）。
2. `.dsh-home/profiles/web/cordis.patch.yml` 追加两行 insert：客户端插件行 + MCP 行。

DSH 会热加载 patch 文件（web profile 关掉了模块级 HMR，但 `cordis.patch.yml` 的
watch-only HMR 是专门为此挂载的），所以**不必重启 dsh**；但新增的浏览器插件行
只在**页面刷新**后进入 `__DSH_BOOT__`——刷新一次即可看到「同传」标签页。

### 回滚

```bash
# 1) 删掉软链
rm "$DSH_HOME/profiles/web/node_modules/dsh-voice-interpreter"

# 2) 删掉 cordis.patch.yml 末尾那段（以标题行开始到文件尾）
#    "── Voice: local transcription + English-to-Chinese live interpretation ──"

# 3) 停掉本地服务
pkill -f "dsh_voice.server"
```

`dsh-voice/` 目录本身、`.venv-audio/`、模型都可原样保留；会议记录在 `dsh-voice/meetings/`，
删掉不影响 DSH 其它功能。

---

## 已知边界

- **服务进程的归属**：插件的 Node 半边在 DSH 启动时探测 `/health`，没服务才拉起一个；
  并每 30 秒巡检一次，崩了会自动重启；它**只杀自己拉起的子进程**，不会动你手动起的服务。
  这个巡检逻辑要**下次启动 dsh 才生效**（宿主插件的 ESM 模块会被缓存）。
  在此之前服务由 `dsh-voice serve` 或 `ensure_live_server` 工具负责。
- **会议记录只在识别成功时才有内容**：没有语音就没有句子，空会议不会产生纪要。
- **纪要质量取决于记录质量**：识别错的专有名词会原样进纪要。用
  `transcribe_audio --initial-prompt` 或 `ingest` 时提供词汇表能显著改善。
- **macOS 权限**：浏览器首次使用需允许麦克风；命令行录音需要给宿主进程麦克风权限。
  模型推理本身不需要任何系统权限。
- **首次翻译/纪要需要联网**；识别与记录全程离线，可以在没网时先录后整理。
- `dsh-voice record` 依赖 ffmpeg 的 `avfoundation` 输入；本机自动使用 `imageio-ffmpeg` 自带的 ffmpeg 7.1。
- `selftest --audio` 之外的模式依赖 macOS `say` 合成语音。新机器的**默认语音可能是音效**
  （Bells、Zarvox 等），会把长文本截断——自检因此显式挑一个真实语音（`Samantha` 等），
  可用 `--voice` 或 `DSH_VOICE_SAY_VOICE` 覆盖。
- 预览行可能先给出一个错的词（例如把 "such short notice" 听成 "Sunday"），
  提交时会被正确答案替换——这是预览的预期行为，不是 bug。
