"""The deployed site fetches its weights from this repo's GitHub release assets
(https://github.com/blazingphoenix7/bitcrush/releases/tag/v1.0). To replace them after a re-export:

    quant/.venv/Scripts/python.exe quant/export_qwen3i.py     # regenerate web/weights-qwen3/
    gh release upload v1.0 web/weights-qwen3/qwen3.bin web/weights-qwen3/manifest.json --clobber

WEIGHTS_REMOTE in web/machine.mjs must point at the release's download base URL.
"""
print(__doc__)
