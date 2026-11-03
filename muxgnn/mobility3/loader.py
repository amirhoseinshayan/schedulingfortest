import json
from pathlib import Path
import torch
from .settings import DATASET_DIR
class Mobility3Dataset:
    def __init__(self,split='train',root=None):
        self.root=Path(root) if root else DATASET_DIR
        manifest=json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        self.paths=[p for name in manifest['splits'][split] for p in sorted((self.root/split/name).rglob('snapshot_*.pt'))]
        if not self.paths:raise FileNotFoundError('No snapshots in selected split: '+split)
    def __len__(self):return len(self.paths)
    def __getitem__(self,i):return torch.load(self.paths[i],map_location='cpu',weights_only=False)
