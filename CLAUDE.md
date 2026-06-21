# CLAUDE.md

## Import Order

Always import `instrumentation` before any `anthropic` import. It must patch the Anthropic SDK before it loads.

```python
import instrumentation
instrumentation.setup()

import anthropic  # after
```
