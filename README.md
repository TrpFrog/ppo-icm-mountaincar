# drl-spring-final

このリポジトリは「DRLスプリングセミナー2023」の最終課題のリポジトリです。

## タスク

このリポジトリでは「MountainCar-v0」の環境を使用し、
[PPO (Proximal Policy Optimization)](https://arxiv.org/abs/1707.06347) + 
[ICM (Intrinsic Curiosity Module)](https://pathak22.github.io/noreward-rl/) による強化学習を行います。

PPO の実装は [nikhilbarhate99/PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch) (MIT License) を参考にしました。

## 動作確認済み環境

  - macOS 13.2.1
  - Python 3.10.8
  - OpenAI Gym 0.26.2
  - PyTorch 2.0.0
 
その他の使用ライブラリについては `requirements.txt` の通りです。

## 動作方法

### 前準備

```bash
pip install -r requirements.txt
wandb disabled
```

### 学習

```bash
python -m src --train --name=NAME
```

`NAME.pth` として学習済みモデルが保存されます。

その他のオプションは `python -m src --help` で確認できます。

### テスト

```bash
python -m src --test --path=PATH_TO_MODEL
```

`PATH_TO_MODEL` として選択可能な学習済みモデルは以下の通りです。

  - `./models/random.pth`: ランダム初期値のモデル
  - `./models/ppo-only.pth`: PPO のみで学習したモデル
  - `./models/ppo-icm.pth`: PPO + ICM で学習したモデル
  - `./models/ppo-icm-pen.pth`: PPO + ICM で学習 + ペナルティを加えたモデル
    - ペナルティ: 200 ステップ経過時点でゴールしていない場合に負の報酬 -100 を与える

`ppo-only`, `ppo-icm`, `ppo-icm-pen` の学習過程は以下のサイトで確認できます。

https://wandb.ai/trpfrog/PPO-ICM-MountainCar

学習に使用したハイパーパラメータは `config.py` の通りです。

