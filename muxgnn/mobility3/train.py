"""Online PPO with frozen 5-relation muxGNN and runtime migration Transformer."""
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from .env import Mobility3Env
from ..rl.policy import MaskedMuxActorCriticPolicy
from ..rl.config import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--timesteps',type=int,default=20000)
    p.add_argument('--encoder',type=Path,default=Path('artifacts/mobility3/encoder.pt'))
    p.add_argument('--output',type=Path,default=Path('artifacts/mobility3/ppo_model.zip'))
    args=p.parse_args()
    if not args.encoder.exists():raise FileNotFoundError('Run mobility3.pretrain first')
    env=Monitor(Mobility3Env(random_scenarios=True))
    policy_kwargs={'hidden_dim':MUX_HIDDEN_DIM,'heads':MUX_HEADS,
        'layers_per_relation':MUX_LAYERS_PER_RELATION,
        'pair_hidden_dim':MUX_PAIR_HIDDEN_DIM,'global_hidden_dim':MUX_GLOBAL_HIDDEN_DIM,
        'actor_hidden_dim':MUX_ACTOR_HIDDEN_DIM,'critic_hidden_dim':MUX_CRITIC_HIDDEN_DIM,
        'dropout':MUX_DROPOUT,'exploration_alpha':0.0,'freeze_graph_encoder':True}
    model=PPO(MaskedMuxActorCriticPolicy,env,learning_rate=MUX_LEARNING_RATE,n_steps=MUX_N_STEPS,
        batch_size=MUX_BATCH_SIZE,n_epochs=MUX_N_EPOCHS,gamma=MUX_GAMMA,
        gae_lambda=MUX_GAE_LAMBDA,clip_range=MUX_CLIP_RANGE,ent_coef=MUX_ENT_COEF,
        vf_coef=MUX_VF_COEF,max_grad_norm=MUX_MAX_GRAD_NORM,policy_kwargs=policy_kwargs)
    model.policy.load_pretrained_encoder(args.encoder)
    model.learn(total_timesteps=args.timesteps)
    args.output.parent.mkdir(parents=True,exist_ok=True);model.save(str(args.output))
    print('Saved',args.output,'with 2560 schedule + 2560 keep/migration actions')
if __name__=='__main__':main()
