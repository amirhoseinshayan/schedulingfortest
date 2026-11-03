"""Same original self-supervised muxGNN pretraining loss; new event graph splits."""
import argparse,json,random,torch,numpy as np
from pathlib import Path
from .settings import DATASET_VERSION
from .loader import Mobility3Dataset
from ..rl.adapter import HeteroMuxAdapter
from ..rl.offline_pretrain import OfflineMuxGNNPretrainer,_run_epoch
from ..rl.config import MUX_HIDDEN_DIM,MUX_HEADS,MUX_LAYERS_PER_RELATION,MUX_GLOBAL_HIDDEN_DIM,MUX_DROPOUT

def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--device',default='auto')
    p.add_argument('--output',type=Path,default=Path('artifacts/mobility3/encoder.pt'))
    args=p.parse_args();device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device)
    train=Mobility3Dataset('train');val=Mobility3Dataset('val')
    adapter=HeteroMuxAdapter()
    net=OfflineMuxGNNPretrainer(adapter.observation_space,MUX_HIDDEN_DIM,MUX_HEADS,MUX_LAYERS_PER_RELATION,MUX_GLOBAL_HIDDEN_DIM,MUX_DROPOUT).to(device)
    opt=torch.optim.Adam(net.parameters(),lr=.001)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    best=float('inf')
    for e in range(args.epochs):
        order=list(range(len(train)));random.Random(e+42).shuffle(order)
        tr=_run_epoch(net,train,adapter,order,device,args.batch_size,.15,.25,1.0,opt)
        va=_run_epoch(net,val,adapter,list(range(len(val))),device,args.batch_size,.15,.25,1.0,None)
        print('epoch',e,'train',tr['loss'],'val',va['loss'])
        if va['loss']<best:
            best=va['loss'];torch.save({'encoder_state_dict':net.encoder.state_dict(),
                'dataset_version':DATASET_VERSION,'derived_time_scale_factor':5.0,'architecture':{
                'hidden_dim':MUX_HIDDEN_DIM,'heads':MUX_HEADS,'layers_per_relation':MUX_LAYERS_PER_RELATION,
                'global_hidden_dim':MUX_GLOBAL_HIDDEN_DIM,'dropout':MUX_DROPOUT},'validation_loss':best},args.output)
if __name__=='__main__':main()
