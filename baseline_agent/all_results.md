# Reference Agent Results

Scores of the reference agent (`baseline_agent/`) on all 118 SciLaws-Bench tasks,
with `max_turns=30`. Real-setting prompts include the input ranges only; Parallel-setting
prompts do not include test ranges. These are the numbers reported in the paper.

Scoring: every value uses strict aggregation over all 118 tasks. A submission that is
missing, `null`, fails to import or execute, violates the contract, or fails the
anti-hacking check contributes `0.0` to its model's score. "Qualified" counts tasks
whose submission passed all of those checks. The `-fg` suffix marks runs made with the
provider's first-party API.

## Overall

| Model | Real numeric ALL | Real typeI | Real typeII | Real qualified | Parallel ALL | Parallel typeI | Parallel typeII | Real validity ALL | Validity typeI | Validity typeII |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gpt-5.5` | 0.5073 | 0.5235 | 0.4866 | 117/118 (99.2%) | 0.5826 | 0.6023 | 0.5577 | 0.8184 | 0.8779 | 0.7429 |
| `gemini-3.5-flash-fg` | 0.4608 | 0.4667 | 0.4534 | 114/118 (96.6%) | 0.5042 | 0.5417 | 0.4567 | 0.7463 | 0.8469 | 0.6186 |
| `glm-5.2` | 0.4584 | 0.4929 | 0.4146 | 110/118 (93.2%) | 0.5212 | 0.5644 | 0.4663 | 0.7419 | 0.8175 | 0.6458 |
| `claude-opus-4-8-fg` | 0.4494 | 0.4571 | 0.4396 | 114/118 (96.6%) | 0.5000 | 0.5303 | 0.4615 | 0.8024 | 0.8803 | 0.7035 |
| `deepseek-v4-pro` | 0.4396 | 0.4221 | 0.4618 | 116/118 (98.3%) | 0.4979 | 0.5114 | 0.4808 | 0.8130 | 0.8375 | 0.7819 |
| `qwen3.7-max` | 0.4380 | 0.4339 | 0.4433 | 107/118 (90.7%) | 0.5042 | 0.5455 | 0.4519 | 0.7266 | 0.7339 | 0.7172 |
| `gpt-5-mini` | 0.4069 | 0.3979 | 0.4182 | 114/118 (96.6%) | 0.4470 | 0.4470 | 0.4471 | 0.7703 | 0.7939 | 0.7403 |
| `gpt5.4-mini` | 0.3717 | 0.3662 | 0.3786 | 113/118 (95.8%) | 0.4258 | 0.4318 | 0.4183 | 0.7245 | 0.7670 | 0.6706 |
| `gpt-4o-mini` | 0.2059 | 0.2727 | 0.1212 | 102/118 (86.4%) | 0.3347 | 0.3447 | 0.3221 | 0.5602 | 0.6817 | 0.4060 |

## Real Numeric Details

| Model | Split | Tasks | Qualified | Numeric score |
|---|---:|---:|---:|---:|
| `gpt-5.5` | ALL | 118 | 117 | 0.5073 |
| `gpt-5.5` | typeI | 66 | 66 | 0.5235 |
| `gpt-5.5` | typeII | 52 | 51 | 0.4866 |
| `gemini-3.5-flash-fg` | ALL | 118 | 114 | 0.4608 |
| `gemini-3.5-flash-fg` | typeI | 66 | 66 | 0.4667 |
| `gemini-3.5-flash-fg` | typeII | 52 | 48 | 0.4534 |
| `glm-5.2` | ALL | 118 | 110 | 0.4584 |
| `glm-5.2` | typeI | 66 | 66 | 0.4929 |
| `glm-5.2` | typeII | 52 | 44 | 0.4146 |
| `claude-opus-4-8-fg` | ALL | 118 | 114 | 0.4494 |
| `claude-opus-4-8-fg` | typeI | 66 | 65 | 0.4571 |
| `claude-opus-4-8-fg` | typeII | 52 | 49 | 0.4396 |
| `deepseek-v4-pro` | ALL | 118 | 116 | 0.4396 |
| `deepseek-v4-pro` | typeI | 66 | 64 | 0.4221 |
| `deepseek-v4-pro` | typeII | 52 | 52 | 0.4618 |
| `qwen3.7-max` | ALL | 118 | 107 | 0.4380 |
| `qwen3.7-max` | typeI | 66 | 60 | 0.4339 |
| `qwen3.7-max` | typeII | 52 | 47 | 0.4433 |
| `gpt-5-mini` | ALL | 118 | 114 | 0.4069 |
| `gpt-5-mini` | typeI | 66 | 65 | 0.3979 |
| `gpt-5-mini` | typeII | 52 | 49 | 0.4182 |
| `gpt5.4-mini` | ALL | 118 | 113 | 0.3717 |
| `gpt5.4-mini` | typeI | 66 | 66 | 0.3662 |
| `gpt5.4-mini` | typeII | 52 | 47 | 0.3786 |
| `gpt-4o-mini` | ALL | 118 | 102 | 0.2059 |
| `gpt-4o-mini` | typeI | 66 | 64 | 0.2727 |
| `gpt-4o-mini` | typeII | 52 | 38 | 0.1212 |

## Parallel Structure Details

| Model | Split | Tasks | Structure score |
|---|---:|---:|---:|
| `gpt-5.5` | ALL | 118 | 0.5826 |
| `gpt-5.5` | typeI | 66 | 0.6023 |
| `gpt-5.5` | typeII | 52 | 0.5577 |
| `glm-5.2` | ALL | 118 | 0.5212 |
| `glm-5.2` | typeI | 66 | 0.5644 |
| `glm-5.2` | typeII | 52 | 0.4663 |
| `gemini-3.5-flash-fg` | ALL | 118 | 0.5042 |
| `gemini-3.5-flash-fg` | typeI | 66 | 0.5417 |
| `gemini-3.5-flash-fg` | typeII | 52 | 0.4567 |
| `qwen3.7-max` | ALL | 118 | 0.5042 |
| `qwen3.7-max` | typeI | 66 | 0.5455 |
| `qwen3.7-max` | typeII | 52 | 0.4519 |
| `claude-opus-4-8-fg` | ALL | 118 | 0.5000 |
| `claude-opus-4-8-fg` | typeI | 66 | 0.5303 |
| `claude-opus-4-8-fg` | typeII | 52 | 0.4615 |
| `deepseek-v4-pro` | ALL | 118 | 0.4979 |
| `deepseek-v4-pro` | typeI | 66 | 0.5114 |
| `deepseek-v4-pro` | typeII | 52 | 0.4808 |
| `gpt-5-mini` | ALL | 118 | 0.4470 |
| `gpt-5-mini` | typeI | 66 | 0.4470 |
| `gpt-5-mini` | typeII | 52 | 0.4471 |
| `gpt5.4-mini` | ALL | 118 | 0.4258 |
| `gpt5.4-mini` | typeI | 66 | 0.4318 |
| `gpt5.4-mini` | typeII | 52 | 0.4183 |
| `gpt-4o-mini` | ALL | 118 | 0.3347 |
| `gpt-4o-mini` | typeI | 66 | 0.3447 |
| `gpt-4o-mini` | typeII | 52 | 0.3221 |

## Real Validity Details

| Model | Split | Tasks | Valid judge results | Anti-hacking fails | Validity score |
|---|---:|---:|---:|---:|---:|
| `gpt-5.5` | ALL | 118 | 117 | 2 | 0.8184 |
| `gpt-5.5` | typeI | 66 | 66 | 1 | 0.8779 |
| `gpt-5.5` | typeII | 52 | 51 | 1 | 0.7429 |
| `deepseek-v4-pro` | ALL | 118 | 116 | 1 | 0.8130 |
| `deepseek-v4-pro` | typeI | 66 | 64 | 1 | 0.8375 |
| `deepseek-v4-pro` | typeII | 52 | 52 | 0 | 0.7819 |
| `claude-opus-4-8-fg` | ALL | 118 | 114 | 1 | 0.8024 |
| `claude-opus-4-8-fg` | typeI | 66 | 65 | 1 | 0.8803 |
| `claude-opus-4-8-fg` | typeII | 52 | 49 | 0 | 0.7035 |
| `gpt-5-mini` | ALL | 118 | 114 | 4 | 0.7703 |
| `gpt-5-mini` | typeI | 66 | 65 | 4 | 0.7939 |
| `gpt-5-mini` | typeII | 52 | 49 | 0 | 0.7403 |
| `gemini-3.5-flash-fg` | ALL | 118 | 114 | 11 | 0.7463 |
| `gemini-3.5-flash-fg` | typeI | 66 | 66 | 4 | 0.8469 |
| `gemini-3.5-flash-fg` | typeII | 52 | 48 | 7 | 0.6186 |
| `glm-5.2` | ALL | 118 | 112 | 6 | 0.7419 |
| `glm-5.2` | typeI | 66 | 66 | 5 | 0.8175 |
| `glm-5.2` | typeII | 52 | 46 | 1 | 0.6458 |
| `qwen3.7-max` | ALL | 118 | 108 | 4 | 0.7266 |
| `qwen3.7-max` | typeI | 66 | 60 | 4 | 0.7339 |
| `qwen3.7-max` | typeII | 52 | 48 | 0 | 0.7172 |
| `gpt5.4-mini` | ALL | 118 | 113 | 5 | 0.7245 |
| `gpt5.4-mini` | typeI | 66 | 66 | 5 | 0.7670 |
| `gpt5.4-mini` | typeII | 52 | 47 | 0 | 0.6706 |
| `gpt-4o-mini` | ALL | 118 | 104 | 2 | 0.5602 |
| `gpt-4o-mini` | typeI | 66 | 64 | 2 | 0.6817 |
| `gpt-4o-mini` | typeII | 52 | 40 | 0 | 0.4060 |
