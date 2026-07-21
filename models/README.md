# Models

Drop the trained model / keyword files here. They are **not** committed to git
(`*.pt` and `*.ppn` are git-ignored because they are large), so each user
supplies their own copies.

| File | Purpose | Config variable |
|---|---|---|
| `my_model.pt` | YOLO scene detection model | `MODEL_KITCHEN` |
| `my_model_spec.pt` | YOLO spectacle detection model | `MODEL_SPEC` |
| `hey-pico.ppn` | Porcupine wake-word ("Hey Pico") | `KEYWORD_PATH` |

After placing the files here, point `.env` at them, e.g.:

```
MODEL_KITCHEN=./models/my_model.pt
MODEL_SPEC=./models/my_model_spec.pt
KEYWORD_PATH=./models/hey-pico.ppn
```

> The `.ppn` wake-word file is platform-specific — use the build that matches
> your device (e.g. Raspberry Pi vs. desktop).
