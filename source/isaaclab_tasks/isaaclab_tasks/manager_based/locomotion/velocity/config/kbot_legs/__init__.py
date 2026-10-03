"""Register Gym environments for the legs-only kbot."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Velocity-Rough-KbotLegs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:KBotLegsRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KBotLegsRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-KbotLegs-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:KBotLegsRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KBotLegsRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Track-KbotLegs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.track_env_cfg:KBotLegsTrackEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KBotLegsTrackPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-TrackMulti-KbotLegs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trackmulti_env_cfg:KBotLegsTrackMultiEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KBotLegsTrackMultiPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-KbotLegs-AMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:KBotLegsAmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KBotLegsAmpPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-TrackClip-KbotLegs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.clip_env_cfg:KBotLegsClipEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KBotLegsClipPPORunnerCfg",
    },
)
