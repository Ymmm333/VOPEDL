# # # #
# # # # import torch
# # # # from torch import nn
# # # # from typing import Callable
# # # # from tqdm import tqdm
# # # # import os
# # # # import json
# # # #
# # # # from utils.Optimizer import OptimizerManager
# # # # from utils.pyExt import dataToDevice, getFunc
# # # # from utils.logger import ProgressLogger
# # # # from utils.typing import Sequence, Collecter, Loader
# # # #
# # # #
# # # # class Trainer:
# # # #     def __init__(self, model: nn.Module, device: torch.device, ckpt_dir: str = "checkpoints"):
# # # #         self.model = model.to(device)
# # # #         self.device = device
# # # #         self.ckpt_dir = ckpt_dir
# # # #         os.makedirs(self.ckpt_dir, exist_ok=True)
# # # #         self.best_hos = 0.0  # 使用 HOS 值作为最佳指标
# # # #
# # # #     def save_checkpoint(self, epoch: int, optimizers, name: str = "last.pth", metrics: dict = None):
# # # #         """保存模型和优化器状态"""
# # # #         if type(optimizers) not in [list, tuple]:
# # # #             optimizers = [optimizers]
# # # #
# # # #         state = {
# # # #             "epoch": epoch,
# # # #             "model": self.model.state_dict(),
# # # #             "optimizers": [opt.state_dict() for opt in optimizers],
# # # #             "best_hos": self.best_hos,  # 保存最佳 HOS 值
# # # #             "metrics": metrics  # 保存评估指标
# # # #         }
# # # #         path = os.path.join(self.ckpt_dir, name)
# # # #         torch.save(state, path)
# # # #         print(f"✅ Checkpoint saved to {path}")
# # # #
# # # #     def load_checkpoint(self, optimizers=None, name: str = "last.pth"):
# # # #         """加载模型和优化器状态"""
# # # #         path = os.path.join(self.ckpt_dir, name)
# # # #         if not os.path.exists(path):
# # # #             print(f"⚠️ No checkpoint found at {path}")
# # # #             return 0  # 从第0个epoch开始
# # # #
# # # #         state = torch.load(path, map_location=self.device)
# # # #         self.model.load_state_dict(state["model"])
# # # #
# # # #         # 加载最佳 HOS 值
# # # #         if "best_hos" in state:
# # # #             self.best_hos = state["best_hos"]
# # # #
# # # #         if optimizers is not None:
# # # #             if type(optimizers) not in [list, tuple]:
# # # #                 optimizers = [optimizers]
# # # #             for opt, opt_state in zip(optimizers, state["optimizers"]):
# # # #                 opt.load_state_dict(opt_state)
# # # #
# # # #         print(f"🔄 Checkpoint loaded from {path}, epoch={state['epoch']}, best_hos={self.best_hos:.4f}")
# # # #
# # # #         # 打印保存的指标
# # # #         if "metrics" in state and state["metrics"]:
# # # #             print("📊 Saved metrics:")
# # # #             for key, value in state["metrics"].items():
# # # #                 if isinstance(value, (int, float)):
# # # #                     print(f"   {key}: {value:.4f}")
# # # #                 elif isinstance(value, list):
# # # #                     print(f"   {key}: {[f'{v:.4f}' for v in value]}")
# # # #                 else:
# # # #                     print(f"   {key}: {value}")
# # # #
# # # #         return state["epoch"]
# # # #
# # # #     def save_best_checkpoint(self, epoch: int, optimizers, current_hos: float, metrics: dict):
# # # #         """保存最佳模型和所有评估指标 - 使用 HOS 值作为评判标准"""
# # # #         if current_hos > self.best_hos:
# # # #             self.best_hos = current_hos
# # # #             if type(optimizers) not in [list, tuple]:
# # # #                 optimizers = [optimizers]
# # # #
# # # #             state = {
# # # #                 "epoch": epoch,
# # # #                 "model": self.model.state_dict(),
# # # #                 "optimizers": [opt.state_dict() for opt in optimizers],
# # # #                 "best_hos": self.best_hos,
# # # #                 "metrics": metrics  # 保存所有评估指标
# # # #             }
# # # #             path = os.path.join(self.ckpt_dir, "best.pth")
# # # #             torch.save(state, path)
# # # #             print(f"🏆 New best model saved! HOS: {current_hos:.4f}")
# # # #
# # # #             # 打印保存的指标
# # # #             print("📊 Best model metrics:")
# # # #             for key, value in metrics.items():
# # # #                 if isinstance(value, (int, float)):
# # # #                     print(f"   {key}: {value:.4f}")
# # # #                 elif isinstance(value, list):
# # # #                     print(f"   {key}: {[f'{v:.4f}' for v in value]}")
# # # #                 else:
# # # #                     print(f"   {key}: {value}")
# # # #
# # # #     def train(self, hook: str, dataloader: Loader, epochs: int, resume: bool = False):
# # # #         self.model.train()
# # # #         progress = ProgressLogger(epochs)
# # # #         self.model.progress = progress
# # # #
# # # #         optimizer = getattr(self.model, f'{hook}_optimizer')()
# # # #         if type(optimizer) not in [list, tuple]:
# # # #             optimizer = [optimizer]
# # # #         loop_step: Callable = getattr(self.model, f'{hook}_step')
# # # #
# # # #         # 安全地获取 epoch_end 函数，如果不存在则返回返回空字典的函数
# # # #         epoch_end_func = getattr(self.model, f'{hook}_epoch_end', lambda: {})
# # # #         if not callable(epoch_end_func):
# # # #             epoch_end_func = lambda: {}
# # # #
# # # #         start_epoch = 0
# # # #         if resume:
# # # #             start_epoch = self.load_checkpoint(optimizers=optimizer)  # 恢复训练
# # # #
# # # #         for epoch in range(start_epoch, epochs):
# # # #             for data in dataloader:
# # # #                 data = dataToDevice(data, self.device)
# # # #                 step_out = loop_step(data)
# # # #                 loss, information = parseTrainStepOut(step_out)
# # # #
# # # #                 with OptimizerManager(optimizer):
# # # #                     loss.backward()
# # # #
# # # #                 progress.add_information(information)
# # # #
# # # #             # 调用 epoch_end 函数并处理可能的异常
# # # #             try:
# # # #                 epoch_out = epoch_end_func()
# # # #             except Exception as e:
# # # #                 print(f"⚠️ Error in {hook}_epoch_end: {e}")
# # # #                 epoch_out = {}
# # # #
# # # #             # 确保 epoch_out 是一个字典
# # # #             if epoch_out is None:
# # # #                 epoch_out = {}
# # # #
# # # #             progress.update(epoch_out)
# # # #
# # # #             # 只有在 epoch_out 是字典时才检查其中的键
# # # #             if isinstance(epoch_out, dict):
# # # #                 # 使用 HOS 值作为评判标准
# # # #                 if 'hos' in epoch_out:
# # # #                     self.save_best_checkpoint(epoch + 1, optimizer, epoch_out['hos'], epoch_out)
# # # #                 # 如果没有 HOS 值，则使用其他指标作为备选
# # # #                 elif 'source_oa' in epoch_out:
# # # #                     print(f"⚠️ No HOS value found, using source_oa as alternative: {epoch_out['source_oa']:.4f}")
# # # #                     # 注意：这里我们不保存最佳模型，因为这不是 HOS 指标
# # # #
# # # #             # 每个epoch保存一次last.pth
# # # #             self.save_checkpoint(epoch + 1, optimizer, name="last.pth", metrics=epoch_out)
# # # #
# # # #         progress.close()
# # # #
# # # #     def test(self, hook: str, dataloader: Loader):
# # # #         self.model.eval()
# # # #
# # # #         loop_step: Callable = getattr(self.model, f'{hook}_step')
# # # #         test_end = getFunc(self.model, f'{hook}_end')
# # # #
# # # #         with torch.no_grad():
# # # #             for data in tqdm(dataloader):
# # # #                 data = dataToDevice(data, self.device)
# # # #                 loop_step(data)
# # # #
# # # #             test_result = test_end()
# # # #
# # # #             # 如果是测试阶段，也保存最佳模型（基于测试集的 HOS 指标）
# # # #             if hook == 'test' and test_result is not None and isinstance(test_result, dict):
# # # #                 # 获取优化器用于保存checkpoint
# # # #                 try:
# # # #                     optimizer = getattr(self.model, 'train_optimizer')()
# # # #                     if type(optimizer) not in [list, tuple]:
# # # #                         optimizer = [optimizer]
# # # #
# # # #                     # 使用 HOS 值作为评判标准
# # # #                     if 'hos' in test_result:
# # # #                         self.save_best_checkpoint(0, optimizer, test_result['hos'], test_result)
# # # #                     else:
# # # #                         print("⚠️ No HOS value found in test results, skipping best model save")
# # # #                 except Exception as e:
# # # #                     print(f"⚠️ Could not save best model after test: {e}")
# # # #
# # # #
# # # # def parseTrainStepOut(step_out: Collecter) -> Sequence:
# # # #     out_type = type(step_out)
# # # #
# # # #     if out_type == dict:
# # # #         loss = step_out['loss']
# # # #         information = step_out['information']
# # # #     elif out_type == list or out_type == tuple:
# # # #         loss = step_out[0]
# # # #         information = step_out[1]
# # # #     else:
# # # #         loss = step_out
# # # #         information = dict(loss=loss)
# # # #
# # # #     return loss, information
# # #
# # # import torch
# # # from torch import nn
# # # from typing import Callable
# # # from tqdm import tqdm
# # # import os
# # # import json
# # # import time
# # #
# # # from utils.Optimizer import OptimizerManager
# # # from utils.pyExt import dataToDevice, getFunc
# # # from utils.logger import ProgressLogger
# # # from utils.typing import Sequence, Collecter, Loader
# # #
# # #
# # # class Trainer:
# # #     def __init__(self, model: nn.Module, device: torch.device, ckpt_dir: str = "checkpoints"):
# # #         self.model = model.to(device)
# # #         self.device = device
# # #         self.ckpt_dir = ckpt_dir
# # #         os.makedirs(self.ckpt_dir, exist_ok=True)
# # #
# # #         # 历史最佳HOS文件路径
# # #         self.history_file = os.path.join(self.ckpt_dir, "history_best.json")
# # #
# # #         # 当前运行最佳HOS文件路径
# # #         self.current_run_file = os.path.join(self.ckpt_dir, "current_run_best.json")
# # #
# # #         # 加载历史最佳HOS和当前运行最佳HOS
# # #         self.best_hos_history = self.load_history_best()
# # #         self.current_run_best = self.load_current_run_best()
# # #
# # #     def load_history_best(self):
# # #         """加载历史最佳HOS值"""
# # #         if os.path.exists(self.history_file):
# # #             with open(self.history_file, 'r') as f:
# # #                 history = json.load(f)
# # #                 print(f"📈 加载历史最佳HOS: {history.get('best_hos', 0):.4f}")
# # #                 return history
# # #         else:
# # #             # 初始化历史记录
# # #             history = {
# # #                 "best_hos": 0.0,
# # #                 "best_epoch": 0,
# # #                 "best_seed": None,
# # #                 "best_metrics": {},
# # #                 "all_runs": []
# # #             }
# # #             return history
# # #
# # #     def load_current_run_best(self):
# # #         """加载当前运行最佳HOS值"""
# # #         if os.path.exists(self.current_run_file):
# # #             with open(self.current_run_file, 'r') as f:
# # #                 current = json.load(f)
# # #                 print(f"📊 加载当前运行最佳HOS: {current.get('best_hos', 0):.4f}")
# # #                 return current
# # #         else:
# # #             # 初始化当前运行记录
# # #             current = {
# # #                 "best_hos": 0.0,
# # #                 "best_epoch": 0,
# # #                 "current_seed": None,
# # #                 "best_metrics": {},
# # #                 "run_start_time": time.time()
# # #             }
# # #             return current
# # #
# # #     def save_history_best(self):
# # #         """保存历史最佳HOS值"""
# # #         with open(self.history_file, 'w') as f:
# # #             json.dump(self.best_hos_history, f, indent=2)
# # #
# # #     def save_current_run_best(self):
# # #         """保存当前运行最佳HOS值"""
# # #         with open(self.current_run_file, 'w') as f:
# # #             json.dump(self.current_run_best, f, indent=2)
# # #
# # #     def save_checkpoint(self, epoch: int, optimizers, name: str = "last.pth", metrics: dict = None):
# # #         """保存模型和优化器状态"""
# # #         if type(optimizers) not in [list, tuple]:
# # #             optimizers = [optimizers]
# # #
# # #         state = {
# # #             "epoch": epoch,
# # #             "model": self.model.state_dict(),
# # #             "optimizers": [opt.state_dict() for opt in optimizers],
# # #             "best_hos_history": self.best_hos_history,  # 保存历史最佳记录
# # #             "current_run_best": self.current_run_best,  # 保存当前运行最佳记录
# # #             "metrics": metrics  # 保存评估指标
# # #         }
# # #         path = os.path.join(self.ckpt_dir, name)
# # #         torch.save(state, path)
# # #         print(f"✅ Checkpoint saved to {path}")
# # #
# # #     def load_checkpoint(self, optimizers=None, name: str = "last.pth"):
# # #         """加载模型和优化器状态"""
# # #         path = os.path.join(self.ckpt_dir, name)
# # #         if not os.path.exists(path):
# # #             print(f"⚠️ No checkpoint found at {path}")
# # #             return 0  # 从第0个epoch开始
# # #
# # #         state = torch.load(path, map_location=self.device)
# # #         self.model.load_state_dict(state["model"])
# # #
# # #         # 加载历史最佳记录和当前运行最佳记录
# # #         if "best_hos_history" in state:
# # #             self.best_hos_history = state["best_hos_history"]
# # #         if "current_run_best" in state:
# # #             self.current_run_best = state["current_run_best"]
# # #
# # #         if optimizers is not None:
# # #             if type(optimizers) not in [list, tuple]:
# # #                 optimizers = [optimizers]
# # #             for opt, opt_state in zip(optimizers, state["optimizers"]):
# # #                 opt.load_state_dict(opt_state)
# # #
# # #         print(f"🔄 Checkpoint loaded from {path}, epoch={state['epoch']}")
# # #         print(f"📈 历史最佳HOS: {self.best_hos_history.get('best_hos', 0):.4f}")
# # #         print(f"📊 当前运行最佳HOS: {self.current_run_best.get('best_hos', 0):.4f}")
# # #
# # #         # 打印保存的指标
# # #         if "metrics" in state and state["metrics"]:
# # #             print("📊 Saved metrics:")
# # #             for key, value in state["metrics"].items():
# # #                 if isinstance(value, (int, float)):
# # #                     print(f"   {key}: {value:.4f}")
# # #                 elif isinstance(value, list):
# # #                     print(f"   {key}: {[f'{v:.4f}' for v in value]}")
# # #                 else:
# # #                     print(f"   {key}: {value}")
# # #
# # #         return state["epoch"]
# # #
# # #     def save_best_checkpoint(self, epoch: int, optimizers, current_hos: float, metrics: dict, seed: int = None):
# # #         """保存最佳模型和所有评估指标 - 同时更新历史最佳和当前运行最佳"""
# # #         # 检查是否是历史最佳
# # #         is_historical_best = current_hos > self.best_hos_history["best_hos"]
# # #
# # #         # 检查是否是当前运行最佳
# # #         is_current_run_best = current_hos > self.current_run_best["best_hos"]
# # #
# # #         # 更新当前运行最佳记录
# # #         if is_current_run_best:
# # #             self.current_run_best["best_hos"] = current_hos
# # #             self.current_run_best["best_epoch"] = epoch
# # #             self.current_run_best["current_seed"] = seed
# # #             self.current_run_best["best_metrics"] = metrics.copy()
# # #             self.save_current_run_best()
# # #
# # #             print(f"🏅 当前运行最佳已更新! HOS: {current_hos:.4f}")
# # #
# # #             # 保存当前运行最佳模型
# # #             if type(optimizers) not in [list, tuple]:
# # #                 optimizers = [optimizers]
# # #
# # #             state = {
# # #                 "epoch": epoch,
# # #                 "model": self.model.state_dict(),
# # #                 "optimizers": [opt.state_dict() for opt in optimizers],
# # #                 "best_hos_history": self.best_hos_history,
# # #                 "current_run_best": self.current_run_best,
# # #                 "metrics": metrics
# # #             }
# # #             path = os.path.join(self.ckpt_dir, "current_best.pth")
# # #             torch.save(state, path)
# # #             print(f"💾 当前运行最佳模型已保存: current_best.pth")
# # #
# # #         # 更新历史最佳记录
# # #         if is_historical_best:
# # #             # 更新历史最佳记录
# # #             self.best_hos_history["best_hos"] = current_hos
# # #             self.best_hos_history["best_epoch"] = epoch
# # #             self.best_hos_history["best_seed"] = seed
# # #             self.best_hos_history["best_metrics"] = metrics.copy()
# # #
# # #             # 添加当前运行记录
# # #             run_record = {
# # #                 "hos": current_hos,
# # #                 "epoch": epoch,
# # #                 "seed": seed,
# # #                 "metrics": metrics.copy(),
# # #                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
# # #             }
# # #             self.best_hos_history["all_runs"].append(run_record)
# # #
# # #             # 只保留最近10次运行记录
# # #             if len(self.best_hos_history["all_runs"]) > 10:
# # #                 self.best_hos_history["all_runs"] = self.best_hos_history["all_runs"][-10:]
# # #
# # #             # 保存历史记录到文件
# # #             self.save_history_best()
# # #
# # #             if type(optimizers) not in [list, tuple]:
# # #                 optimizers = [optimizers]
# # #
# # #             state = {
# # #                 "epoch": epoch,
# # #                 "model": self.model.state_dict(),
# # #                 "optimizers": [opt.state_dict() for opt in optimizers],
# # #                 "best_hos_history": self.best_hos_history,
# # #                 "current_run_best": self.current_run_best,
# # #                 "metrics": metrics
# # #             }
# # #             path = os.path.join(self.ckpt_dir, "best.pth")
# # #             torch.save(state, path)
# # #             print(f"🏆 历史最佳模型已保存! HOS: {current_hos:.4f} (历史最佳)")
# # #
# # #             # 打印保存的指标
# # #             print("📊 历史最佳模型指标:")
# # #             for key, value in metrics.items():
# # #                 if isinstance(value, (int, float)):
# # #                     print(f"   {key}: {value:.4f}")
# # #                 elif isinstance(value, list):
# # #                     print(f"   {key}: {[f'{v:.4f}' for v in value]}")
# # #                 else:
# # #                     print(f"   {key}: {value}")
# # #         else:
# # #             print(
# # #                 f"📊 当前HOS: {current_hos:.4f}, 历史最佳: {self.best_hos_history['best_hos']:.4f}, 当前运行最佳: {self.current_run_best['best_hos']:.4f}")
# # #
# # #     def train(self, hook: str, dataloader: Loader, epochs: int, resume: bool = False, seed: int = None):
# # #         self.model.train()
# # #         progress = ProgressLogger(epochs)
# # #         self.model.progress = progress
# # #
# # #         optimizer = getattr(self.model, f'{hook}_optimizer')()
# # #         if type(optimizer) not in [list, tuple]:
# # #             optimizer = [optimizer]
# # #         loop_step: Callable = getattr(self.model, f'{hook}_step')
# # #
# # #         # 安全地获取 epoch_end 函数，如果不存在则返回返回空字典的函数
# # #         epoch_end_func = getattr(self.model, f'{hook}_epoch_end', lambda: {})
# # #         if not callable(epoch_end_func):
# # #             epoch_end_func = lambda: {}
# # #
# # #         start_epoch = 0
# # #         if resume:
# # #             start_epoch = self.load_checkpoint(optimizers=optimizer)  # 恢复训练
# # #
# # #         # 初始化当前运行记录（如果是新运行）
# # #         if not resume:
# # #             self.current_run_best = {
# # #                 "best_hos": 0.0,
# # #                 "best_epoch": 0,
# # #                 "current_seed": seed,
# # #                 "best_metrics": {},
# # #                 "run_start_time": time.time()
# # #             }
# # #             self.save_current_run_best()
# # #
# # #         for epoch in range(start_epoch, epochs):
# # #             for data in dataloader:
# # #                 data = dataToDevice(data, self.device)
# # #                 step_out = loop_step(data)
# # #                 loss, information = parseTrainStepOut(step_out)
# # #
# # #                 with OptimizerManager(optimizer):
# # #                     loss.backward()
# # #
# # #                 progress.add_information(information)
# # #
# # #             # 调用 epoch_end 函数并处理可能的异常
# # #             try:
# # #                 epoch_out = epoch_end_func()
# # #             except Exception as e:
# # #                 print(f"⚠️ Error in {hook}_epoch_end: {e}")
# # #                 epoch_out = {}
# # #
# # #             # 确保 epoch_out 是一个字典
# # #             if epoch_out is None:
# # #                 epoch_out = {}
# # #
# # #             progress.update(epoch_out)
# # #
# # #             # 只有在 epoch_out 是字典时才检查其中的键
# # #             if isinstance(epoch_out, dict):
# # #                 # 使用 HOS 值作为评判标准
# # #                 if 'hos' in epoch_out:
# # #                     self.save_best_checkpoint(epoch + 1, optimizer, epoch_out['hos'], epoch_out, seed)
# # #                 # 如果没有 HOS 值，则使用其他指标作为备选
# # #                 elif 'source_oa' in epoch_out:
# # #                     print(f"⚠️ No HOS value found, using source_oa as alternative: {epoch_out['source_oa']:.4f}")
# # #                     # 注意：这里我们不保存最佳模型，因为这不是 HOS 指标
# # #
# # #             # 每个epoch保存一次last.pth
# # #             self.save_checkpoint(epoch + 1, optimizer, name="last.pth", metrics=epoch_out)
# # #
# # #         progress.close()
# # #
# # #     def test(self, hook: str, dataloader: Loader, seed: int = None):
# # #         self.model.eval()
# # #
# # #         loop_step: Callable = getattr(self.model, f'{hook}_step')
# # #         test_end = getFunc(self.model, f'{hook}_end')
# # #
# # #         with torch.no_grad():
# # #             for data in tqdm(dataloader):
# # #                 data = dataToDevice(data, self.device)
# # #                 loop_step(data)
# # #
# # #             test_result = test_end()
# # #
# # #             # 如果是测试阶段，也保存最佳模型（基于测试集的 HOS 指标）
# # #             if hook == 'test' and test_result is not None and isinstance(test_result, dict) and test_result:
# # #                 # 获取优化器用于保存checkpoint
# # #                 try:
# # #                     optimizer = getattr(self.model, 'train_optimizer')()
# # #                     if type(optimizer) not in [list, tuple]:
# # #                         optimizer = [optimizer]
# # #
# # #                     # 使用 HOS 值作为评判标准
# # #                     if 'hos' in test_result:
# # #                         self.save_best_checkpoint(0, optimizer, test_result['hos'], test_result, seed)
# # #                     else:
# # #                         print("⚠️ No HOS value found in test results, skipping best model save")
# # #                 except Exception as e:
# # #                     print(f"⚠️ Could not save best model after test: {e}")
# # #             elif hook == 'test':
# # #                 print("⚠️ Test results are empty or invalid, skipping best model save")
# # #
# # #     def print_history_summary(self):
# # #         """打印历史最佳记录摘要"""
# # #         print(f"\n{'=' * 60}")
# # #         print("📈 历史最佳记录摘要")
# # #         print(f"{'=' * 60}")
# # #         print(f"历史最佳HOS: {self.best_hos_history['best_hos']:.4f}")
# # #         print(f"最佳epoch: {self.best_hos_history['best_epoch']}")
# # #         print(f"最佳seed: {self.best_hos_history['best_seed']}")
# # #         print(f"总运行次数: {len(self.best_hos_history['all_runs'])}")
# # #
# # #         if self.best_hos_history['all_runs']:
# # #             print(f"\n最近运行记录:")
# # #             for i, run in enumerate(self.best_hos_history['all_runs'][-5:]):  # 只显示最近5次
# # #                 print(f"  运行 {i + 1}: HOS={run['hos']:.4f}, seed={run['seed']}, epoch={run['epoch']}")
# # #
# # #         print(f"\n📊 当前运行最佳记录:")
# # #         print(f"当前运行最佳HOS: {self.current_run_best['best_hos']:.4f}")
# # #         print(f"当前运行最佳epoch: {self.current_run_best['best_epoch']}")
# # #         print(f"当前运行seed: {self.current_run_best['current_seed']}")
# # #
# # #         # 计算运行时间
# # #         if 'run_start_time' in self.current_run_best:
# # #             run_time = time.time() - self.current_run_best['run_start_time']
# # #             print(f"当前运行时间: {run_time:.2f}秒")
# # #
# # #         print(f"{'=' * 60}")
# # #
# # #
# # # def parseTrainStepOut(step_out: Collecter) -> Sequence:
# # #     out_type = type(step_out)
# # #
# # #     if out_type == dict:
# # #         loss = step_out['loss']
# # #         information = step_out['information']
# # #     elif out_type == list or out_type == tuple:
# # #         loss = step_out[0]
# # #         information = step_out[1]
# # #     else:
# # #         loss = step_out
# # #         information = dict(loss=loss)
# # #
# # #     return loss, information
# # #
#
#
# import torch
# from torch import nn
# from typing import Callable
# from tqdm import tqdm
# import os
# import json
# import time
#
# from utils.Optimizer import OptimizerManager
# from utils.pyExt import dataToDevice, getFunc
# from utils.logger import ProgressLogger
# from utils.typing import Sequence, Collecter, Loader
#
#
# class Trainer:
#     def __init__(self, model: nn.Module, device: torch.device, ckpt_dir: str = "checkpoints"):
#         self.model = model.to(device)
#         self.device = device
#         self.ckpt_dir = ckpt_dir
#         os.makedirs(self.ckpt_dir, exist_ok=True)
#
#         # 历史最佳HOS文件路径
#         self.history_file = os.path.join(self.ckpt_dir, "history_best.json")
#
#         # 当前运行最佳HOS文件路径
#         self.current_run_file = os.path.join(self.ckpt_dir, "current_run_best.json")
#
#         # 加载历史最佳HOS和当前运行最佳HOS
#         self.best_hos_history = self.load_history_best()
#         self.current_run_best = self.load_current_run_best()
#
#     def load_history_best(self):
#         """加载历史最佳HOS值"""
#         if os.path.exists(self.history_file):
#             with open(self.history_file, 'r') as f:
#                 history = json.load(f)
#                 print(f"📈 加载历史最佳HOS: {history.get('best_hos', 0):.4f}")
#                 return history
#         else:
#             # 初始化历史记录
#             history = {
#                 "best_hos": 0.0,
#                 "best_epoch": 0,
#                 "best_seed": None,
#                 "best_metrics": {},
#                 "all_runs": []
#             }
#             return history
#
#     def load_current_run_best(self):
#         """加载当前运行最佳HOS值"""
#         if os.path.exists(self.current_run_file):
#             with open(self.current_run_file, 'r') as f:
#                 current = json.load(f)
#                 print(f"📊 加载当前运行最佳HOS: {current.get('best_hos', 0):.4f}")
#                 return current
#         else:
#             # 初始化当前运行记录
#             current = {
#                 "best_hos": 0.0,
#                 "best_epoch": 0,
#                 "current_seed": None,
#                 "best_metrics": {},
#                 "run_start_time": time.time()
#             }
#             return current
#
#     def save_history_best(self):
#         """保存历史最佳HOS值"""
#         # 将Tensor转换为Python原生类型
#         serializable_history = self._convert_tensors_to_python(self.best_hos_history)
#         with open(self.history_file, 'w') as f:
#             json.dump(serializable_history, f, indent=2)
#
#     def save_current_run_best(self):
#         """保存当前运行最佳HOS值"""
#         # 将Tensor转换为Python原生类型
#         serializable_current = self._convert_tensors_to_python(self.current_run_best)
#         with open(self.current_run_file, 'w') as f:
#             json.dump(serializable_current, f, indent=2)
#
#     def _convert_tensors_to_python(self, obj):
#         """递归将Tensor转换为Python原生类型"""
#         if isinstance(obj, torch.Tensor):
#             # 如果是单个元素的Tensor，返回其值
#             if obj.numel() == 1:
#                 return obj.item()
#             # 如果是多个元素的Tensor，转换为列表
#             else:
#                 return obj.tolist()
#         elif isinstance(obj, dict):
#             return {k: self._convert_tensors_to_python(v) for k, v in obj.items()}
#         elif isinstance(obj, list):
#             return [self._convert_tensors_to_python(v) for v in obj]
#         else:
#             return obj
#
#     def save_checkpoint(self, epoch: int, optimizers, name: str = "last.pth", metrics: dict = None):
#         """保存模型和优化器状态"""
#         if type(optimizers) not in [list, tuple]:
#             optimizers = [optimizers]
#
#         # 将metrics中的Tensor转换为Python原生类型
#         serializable_metrics = self._convert_tensors_to_python(metrics) if metrics else None
#
#         state = {
#             "epoch": epoch,
#             "model": self.model.state_dict(),
#             "optimizers": [opt.state_dict() for opt in optimizers],
#             "best_hos_history": self.best_hos_history,  # 保存历史最佳记录
#             "current_run_best": self.current_run_best,  # 保存当前运行最佳记录
#             "metrics": serializable_metrics  # 保存评估指标
#         }
#         path = os.path.join(self.ckpt_dir, name)
#         torch.save(state, path)
#         print(f"✅ Checkpoint saved to {path}")
#
#     def load_checkpoint(self, optimizers=None, name: str = "last.pth"):
#         """加载模型和优化器状态"""
#         path = os.path.join(self.ckpt_dir, name)
#         if not os.path.exists(path):
#             print(f"⚠️ No checkpoint found at {path}")
#             return 0  # 从第0个epoch开始
#
#         state = torch.load(path, map_location=self.device)
#         self.model.load_state_dict(state["model"])
#
#         # 加载历史最佳记录和当前运行最佳记录
#         if "best_hos_history" in state:
#             self.best_hos_history = state["best_hos_history"]
#         if "current_run_best" in state:
#             self.current_run_best = state["current_run_best"]
#
#         if optimizers is not None:
#             if type(optimizers) not in [list, tuple]:
#                 optimizers = [optimizers]
#             for opt, opt_state in zip(optimizers, state["optimizers"]):
#                 opt.load_state_dict(opt_state)
#
#         print(f"🔄 Checkpoint loaded from {path}, epoch={state['epoch']}")
#         print(f"📈 历史最佳HOS: {self.best_hos_history.get('best_hos', 0):.4f}")
#         print(f"📊 当前运行最佳HOS: {self.current_run_best.get('best_hos', 0):.4f}")
#
#         # 打印保存的指标
#         if "metrics" in state and state["metrics"]:
#             print("📊 Saved metrics:")
#             for key, value in state["metrics"].items():
#                 if isinstance(value, (int, float)):
#                     print(f"   {key}: {value:.4f}")
#                 elif isinstance(value, list):
#                     print(f"   {key}: {[f'{v:.4f}' for v in value]}")
#                 else:
#                     print(f"   {key}: {value}")
#
#         return state["epoch"]
#
#     def save_best_checkpoint(self, epoch: int, optimizers, current_metric: float, metrics: dict, seed: int = None,
#                              metric_type: str = "hos"):
#         """保存最佳模型和所有评估指标 - 同时更新历史最佳和当前运行最佳"""
#         # 确保current_metric是Python原生类型
#         if isinstance(current_metric, torch.Tensor):
#             current_metric = current_metric.item()
#
#         # 检查是否是历史最佳
#         is_historical_best = current_metric > self.best_hos_history["best_hos"]
#
#         # 检查是否是当前运行最佳
#         is_current_run_best = current_metric > self.current_run_best["best_hos"]
#
#         # 更新当前运行最佳记录
#         if is_current_run_best:
#             self.current_run_best["best_hos"] = current_metric
#             self.current_run_best["best_epoch"] = epoch
#             self.current_run_best["current_seed"] = seed
#             self.current_run_best["best_metrics"] = self._convert_tensors_to_python(metrics)
#             self.current_run_best["metric_type"] = metric_type  # 记录使用的指标类型
#             self.save_current_run_best()
#
#             print(f"🏅 当前运行最佳已更新! {metric_type.upper()}: {current_metric:.4f}")
#
#             # 保存当前运行最佳模型
#             if type(optimizers) not in [list, tuple]:
#                 optimizers = [optimizers]
#
#             state = {
#                 "epoch": epoch,
#                 "model": self.model.state_dict(),
#                 "optimizers": [opt.state_dict() for opt in optimizers],
#                 "best_hos_history": self.best_hos_history,
#                 "current_run_best": self.current_run_best,
#                 "metrics": metrics
#             }
#             path = os.path.join(self.ckpt_dir, "current_best.pth")
#             torch.save(state, path)
#             print(f"💾 当前运行最佳模型已保存: current_best.pth")
#
#         # 更新历史最佳记录
#         if is_historical_best:
#             # 更新历史最佳记录
#             self.best_hos_history["best_hos"] = current_metric
#             self.best_hos_history["best_epoch"] = epoch
#             self.best_hos_history["best_seed"] = seed
#             self.best_hos_history["best_metrics"] = self._convert_tensors_to_python(metrics)
#             self.best_hos_history["metric_type"] = metric_type  # 记录使用的指标类型
#
#             # 添加当前运行记录
#             run_record = {
#                 metric_type: current_metric,
#                 "epoch": epoch,
#                 "seed": seed,
#                 "metrics": self._convert_tensors_to_python(metrics),
#                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#                 "metric_type": metric_type
#             }
#             self.best_hos_history["all_runs"].append(run_record)
#
#             # 只保留最近10次运行记录
#             if len(self.best_hos_history["all_runs"]) > 10:
#                 self.best_hos_history["all_runs"] = self.best_hos_history["all_runs"][-10:]
#
#             # 保存历史记录到文件
#             self.save_history_best()
#
#             if type(optimizers) not in [list, tuple]:
#                 optimizers = [optimizers]
#
#             state = {
#                 "epoch": epoch,
#                 "model": self.model.state_dict(),
#                 "optimizers": [opt.state_dict() for opt in optimizers],
#                 "best_hos_history": self.best_hos_history,
#                 "current_run_best": self.current_run_best,
#                 "metrics": metrics
#             }
#             path = os.path.join(self.ckpt_dir, "best.pth")
#             torch.save(state, path)
#             print(f"🏆 历史最佳模型已保存! {metric_type.upper()}: {current_metric:.4f} (历史最佳)")
#
#             # 打印保存的指标
#             print("📊 历史最佳模型指标:")
#             for key, value in metrics.items():
#                 if isinstance(value, (int, float)) or (isinstance(value, torch.Tensor) and value.numel() == 1):
#                     if isinstance(value, torch.Tensor):
#                         value = value.item()
#                     print(f"   {key}: {value:.4f}")
#                 elif isinstance(value, list) or (isinstance(value, torch.Tensor) and value.numel() > 1):
#                     if isinstance(value, torch.Tensor):
#                         value = value.tolist()
#                     print(f"   {key}: {[f'{v:.4f}' for v in value]}")
#                 else:
#                     print(f"   {key}: {value}")
#         else:
#             print(
#                 f"📊 当前{metric_type.upper()}: {current_metric:.4f}, 历史最佳: {self.best_hos_history['best_hos']:.4f}, 当前运行最佳: {self.current_run_best['best_hos']:.4f}")
#
#     def train(self, hook: str, dataloader: Loader, epochs: int, resume: bool = False, seed: int = None):
#         self.model.train()
#         progress = ProgressLogger(epochs)
#         self.model.progress = progress
#
#         optimizer = getattr(self.model, f'{hook}_optimizer')()
#         if type(optimizer) not in [list, tuple]:
#             optimizer = [optimizer]
#         loop_step: Callable = getattr(self.model, f'{hook}_step')
#
#         # 安全地获取 epoch_end 函数，如果不存在则返回返回空字典的函数
#         epoch_end_func = getattr(self.model, f'{hook}_epoch_end', lambda: {})
#         if not callable(epoch_end_func):
#             epoch_end_func = lambda: {}
#
#         start_epoch = 0
#         if resume:
#             start_epoch = self.load_checkpoint(optimizers=optimizer)  # 恢复训练
#
#         # 初始化当前运行记录（如果是新运行）
#         if not resume:
#             self.current_run_best = {
#                 "best_hos": 0.0,
#                 "best_epoch": 0,
#                 "current_seed": seed,
#                 "best_metrics": {},
#                 "run_start_time": time.time()
#             }
#             self.save_current_run_best()
#
#         for epoch in range(start_epoch, epochs):
#             for data in dataloader:
#                 data = dataToDevice(data, self.device)
#                 step_out = loop_step(data)
#                 loss, information = parseTrainStepOut(step_out)
#
#                 with OptimizerManager(optimizer):
#                     loss.backward()
#
#                 progress.add_information(information)
#
#             # 调用 epoch_end 函数并处理可能的异常
#             try:
#                 epoch_out = epoch_end_func()
#             except Exception as e:
#                 print(f"⚠️ Error in {hook}_epoch_end: {e}")
#                 epoch_out = {}
#
#             # 调用 epoch_end 函数并处理可能的异常
#             try:
#                 epoch_out = epoch_end_func()
#                 # 🔍 添加调试信息：打印 epoch_out 的内容
#                 print(f"🔍 DEBUG: epoch_out 类型: {type(epoch_out)}")
#                 if isinstance(epoch_out, dict):
#                     print(f"🔍 DEBUG: epoch_out 包含的键: {list(epoch_out.keys())}")
#                     for key, value in epoch_out.items():
#                         value_type = type(value)
#                         if hasattr(value, 'shape'):
#                             print(f"   {key}: {value_type} {value.shape}")
#                         elif isinstance(value, (int, float)):
#                             print(f"   {key}: {value}")
#                         else:
#                             print(f"   {key}: {value_type}")
#                 else:
#                     print(f"🔍 DEBUG: epoch_out 不是字典，实际是: {epoch_out}")
#             except Exception as e:
#                 print(f"⚠️ Error in {hook}_epoch_end: {e}")
#                 epoch_out = {}
#
#             # 确保 epoch_out 是一个字典
#             if epoch_out is None:
#                 epoch_out = {}
#
#             progress.update(epoch_out)
#
#             # 只有在 epoch_out 是字典时才检查其中的键
#             if isinstance(epoch_out, dict):
#                 # 使用 HOS 值作为评判标准
#                 if 'hos' in epoch_out:
#                     # 确保hos值是Python原生类型
#                     hos_value = epoch_out['hos']
#                     if isinstance(hos_value, torch.Tensor):
#                         hos_value = hos_value.item()
#                     self.save_best_checkpoint(epoch + 1, optimizer, hos_value, epoch_out, seed, "hos")
#                 # 如果没有 HOS 值，则使用source_oa作为替代
#                 elif 'source_oa' in epoch_out:
#                     # 确保source_oa值是Python原生类型
#                     source_oa_value = epoch_out['source_oa']
#                     if isinstance(source_oa_value, torch.Tensor):
#                         source_oa_value = source_oa_value.item()
#                     self.save_best_checkpoint(epoch + 1, optimizer, source_oa_value, epoch_out, seed, "source_oa")
#                 else:
#                     print(f"⚠️ No HOS or source_oa value found in epoch output")
#
#             # 每个epoch保存一次last.pth
#             self.save_checkpoint(epoch + 1, optimizer, name="last.pth", metrics=epoch_out)
#
#         progress.close()
#
#     def test(self, hook: str, dataloader: Loader, seed: int = None):
#         self.model.eval()
#
#         loop_step: Callable = getattr(self.model, f'{hook}_step')
#         test_end = getFunc(self.model, f'{hook}_end')
#
#         with torch.no_grad():
#             for data in tqdm(dataloader):
#                 data = dataToDevice(data, self.device)
#                 loop_step(data)
#
#             test_result = test_end()
#
#             # 如果是测试阶段，也保存最佳模型（基于测试集的 HOS 指标）
#             if hook == 'test' and test_result is not None and isinstance(test_result, dict) and test_result:
#                 # 获取优化器用于保存checkpoint
#                 try:
#                     optimizer = getattr(self.model, 'train_optimizer')()
#                     if type(optimizer) not in [list, tuple]:
#                         optimizer = [optimizer]
#
#                     # 使用 HOS 值作为评判标准
#                     if 'hos' in test_result:
#                         # 确保hos值是Python原生类型
#                         hos_value = test_result['hos']
#                         if isinstance(hos_value, torch.Tensor):
#                             hos_value = hos_value.item()
#                         self.save_best_checkpoint(0, optimizer, hos_value, test_result, seed, "hos")
#                     else:
#                         print("⚠️ No HOS value found in test results, skipping best model save")
#                 except Exception as e:
#                     print(f"⚠️ Could not save best model after test: {e}")
#             elif hook == 'test':
#                 print("⚠️ Test results are empty or invalid, skipping best model save")
#
#     def print_history_summary(self):
#         """打印历史最佳记录摘要"""
#         print(f"\n{'=' * 60}")
#         print("📈 历史最佳记录摘要")
#         print(f"{'=' * 60}")
#         metric_type = self.best_hos_history.get("metric_type", "hos")
#         print(f"历史最佳{metric_type.upper()}: {self.best_hos_history['best_hos']:.4f}")
#         print(f"最佳epoch: {self.best_hos_history['best_epoch']}")
#         print(f"最佳seed: {self.best_hos_history['best_seed']}")
#         print(f"总运行次数: {len(self.best_hos_history['all_runs'])}")
#
#         if self.best_hos_history['all_runs']:
#             print(f"\n最近运行记录:")
#             for i, run in enumerate(self.best_hos_history['all_runs'][-5:]):  # 只显示最近5次
#                 metric_type = run.get("metric_type", "hos")
#                 metric_value = run.get(metric_type, 0)
#                 print(
#                     f"  运行 {i + 1}: {metric_type.upper()}={metric_value:.4f}, seed={run['seed']}, epoch={run['epoch']}")
#
#         print(f"\n📊 当前运行最佳记录:")
#         current_metric_type = self.current_run_best.get("metric_type", "hos")
#         print(f"当前运行最佳{current_metric_type.upper()}: {self.current_run_best['best_hos']:.4f}")
#         print(f"当前运行最佳epoch: {self.current_run_best['best_epoch']}")
#         print(f"当前运行seed: {self.current_run_best['current_seed']}")
#
#         # 计算运行时间
#         if 'run_start_time' in self.current_run_best:
#             run_time = time.time() - self.current_run_best['run_start_time']
#             print(f"当前运行时间: {run_time:.2f}秒")
#
#         print(f"{'=' * 60}")
#
#
# def parseTrainStepOut(step_out: Collecter) -> Sequence:
#     out_type = type(step_out)
#
#     if out_type == dict:
#         loss = step_out['loss']
#         information = step_out['information']
#     elif out_type == list or out_type == tuple:
#         loss = step_out[0]
#         information = step_out[1]
#     else:
#         loss = step_out
#         information = dict(loss=loss)
#
#     return loss, information
import torch
from torch import nn
from typing import Callable
from tqdm import tqdm
import os
import json
import time

from utils.Optimizer import OptimizerManager
from utils.pyExt import dataToDevice, getFunc
from utils.logger import ProgressLogger
from utils.typing import Sequence, Collecter, Loader


class Trainer:
    def __init__(self, model: nn.Module, device: torch.device, ckpt_dir: str = "checkpoints"):
        self.model = model.to(device)
        self.device = device
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # 历史最佳HOS文件路径
        self.history_file = os.path.join(self.ckpt_dir, "history_best.json")

        # 当前运行最佳HOS文件路径
        self.current_run_file = os.path.join(self.ckpt_dir, "current_run_best.json")

        # 加载历史最佳HOS和当前运行最佳HOS
        self.best_hos_history = self.load_history_best()
        self.current_run_best = self.load_current_run_best()

        # 定义需要检查的关键指标
        self.required_metrics = ['oa', 'aa', 'classes_acc', 'oa_known', 'aa_known', 'unknown', 'hos']

    def load_history_best(self):
        """加载历史最佳HOS值"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                history = json.load(f)
                print(f"📈 加载历史最佳HOS: {history.get('best_hos', 0):.4f}")
                return history
        else:
            # 初始化历史记录
            history = {
                "best_hos": 0.0,
                "best_epoch": 0,
                "best_seed": None,
                "best_metrics": {},
                "all_runs": []
            }
            return history

    def load_current_run_best(self):
        """加载当前运行最佳HOS值"""
        if os.path.exists(self.current_run_file):
            with open(self.current_run_file, 'r') as f:
                current = json.load(f)
                print(f"📊 加载当前运行最佳HOS: {current.get('best_hos', 0):.4f}")
                return current
        else:
            # 初始化当前运行记录
            current = {
                "best_hos": 0.0,
                "best_epoch": 0,
                "current_seed": None,
                "best_metrics": {},
                "run_start_time": time.time()
            }
            return current

    def save_history_best(self):
        """保存历史最佳HOS值"""
        # 将Tensor转换为Python原生类型
        serializable_history = self._convert_tensors_to_python(self.best_hos_history)
        with open(self.history_file, 'w') as f:
            json.dump(serializable_history, f, indent=2)

    def save_current_run_best(self):
        """保存当前运行最佳HOS值"""
        # 将Tensor转换为Python原生类型
        serializable_current = self._convert_tensors_to_python(self.current_run_best)
        with open(self.current_run_file, 'w') as f:
            json.dump(serializable_current, f, indent=2)

    def _convert_tensors_to_python(self, obj):
        """递归将Tensor转换为Python原生类型"""
        if isinstance(obj, torch.Tensor):
            # 如果是单个元素的Tensor，返回其值
            if obj.numel() == 1:
                return obj.item()
            # 如果是多个元素的Tensor，转换为列表
            else:
                return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_tensors_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_tensors_to_python(v) for v in obj]
        else:
            return obj

    def save_checkpoint(self, epoch: int, optimizers, name: str = "last.pth", metrics: dict = None):
        """保存模型和优化器状态"""
        if type(optimizers) not in [list, tuple]:
            optimizers = [optimizers]

        # 将metrics中的Tensor转换为Python原生类型
        serializable_metrics = self._convert_tensors_to_python(metrics) if metrics else None

        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizers": [opt.state_dict() for opt in optimizers],
            "best_hos_history": self.best_hos_history,  # 保存历史最佳记录
            "current_run_best": self.current_run_best,  # 保存当前运行最佳记录
            "metrics": serializable_metrics  # 保存评估指标
        }
        path = os.path.join(self.ckpt_dir, name)
        torch.save(state, path)
        print(f"✅ Checkpoint saved to {path}")

    def load_checkpoint(self, optimizers=None, name: str = "last.pth"):
        """加载模型和优化器状态"""
        path = os.path.join(self.ckpt_dir, name)
        if not os.path.exists(path):
            print(f"⚠️ No checkpoint found at {path}")
            return 0  # 从第0个epoch开始

        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"])

        # 加载历史最佳记录和当前运行最佳记录
        if "best_hos_history" in state:
            self.best_hos_history = state["best_hos_history"]
        if "current_run_best" in state:
            self.current_run_best = state["current_run_best"]

        if optimizers is not None:
            if type(optimizers) not in [list, tuple]:
                optimizers = [optimizers]
            for opt, opt_state in zip(optimizers, state["optimizers"]):
                opt.load_state_dict(opt_state)

        print(f"🔄 Checkpoint loaded from {path}, epoch={state['epoch']}")
        print(f"📈 历史最佳HOS: {self.best_hos_history.get('best_hos', 0):.4f}")
        print(f"📊 当前运行最佳HOS: {self.current_run_best.get('best_hos', 0):.4f}")

        # 打印保存的指标
        if "metrics" in state and state["metrics"]:
            print("📊 Saved metrics:")
            for key, value in state["metrics"].items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.4f}")
                elif isinstance(value, list):
                    print(f"   {key}: {[f'{v:.4f}' for v in value]}")
                else:
                    print(f"   {key}: {value}")

        return state["epoch"]

    def check_required_metrics(self, metrics: dict, mode='test'):
        """根据不同阶段检查不同的必需指标"""
        if mode == 'train':
            # 训练阶段无需检查这些评估指标
            return True
        else:
            # 测试/验证阶段才检查完整的评估指标集
            missing = [metric for metric in self.required_metrics if metric not in metrics]
            if missing:
                print(f"⚠️ 警告: 测试结果缺少以下指标: {missing}")
                return False
            return True

    def save_best_checkpoint(self, epoch: int, optimizers, current_metric: float, metrics: dict, seed: int = None,
                             metric_type: str = "hos"):
        """保存最佳模型和所有评估指标 - 同时更新历史最佳和当前运行最佳"""
        # 确保current_metric是Python原生类型
        if isinstance(current_metric, torch.Tensor):
            current_metric = current_metric.item()

        # 检查必需的指标
        has_required_metrics = self.check_required_metrics(metrics)
        if not has_required_metrics:
            print("⚠️ 缺少必需指标，但仍会继续保存检查点")

        # 检查是否是历史最佳
        is_historical_best = current_metric > self.best_hos_history["best_hos"]

        # 检查是否是当前运行最佳
        is_current_run_best = current_metric > self.current_run_best["best_hos"]

        # 更新当前运行最佳记录
        if is_current_run_best:
            self.current_run_best["best_hos"] = current_metric
            self.current_run_best["best_epoch"] = epoch
            self.current_run_best["current_seed"] = seed
            self.current_run_best["best_metrics"] = self._convert_tensors_to_python(metrics)
            self.current_run_best["metric_type"] = metric_type  # 记录使用的指标类型
            self.save_current_run_best()

            print(f"🏅 当前运行最佳已更新! {metric_type.upper()}: {current_metric:.4f}")

            # 保存当前运行最佳模型
            if type(optimizers) not in [list, tuple]:
                optimizers = [optimizers]

            state = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizers": [opt.state_dict() for opt in optimizers],
                "best_hos_history": self.best_hos_history,
                "current_run_best": self.current_run_best,
                "metrics": metrics
            }
            path = os.path.join(self.ckpt_dir, "current_best.pth")
            torch.save(state, path)
            print(f"💾 当前运行最佳模型已保存: current_best.pth")

        # 更新历史最佳记录
        if is_historical_best:
            # 更新历史最佳记录
            self.best_hos_history["best_hos"] = current_metric
            self.best_hos_history["best_epoch"] = epoch
            self.best_hos_history["best_seed"] = seed
            self.best_hos_history["best_metrics"] = self._convert_tensors_to_python(metrics)
            self.best_hos_history["metric_type"] = metric_type  # 记录使用的指标类型

            # 添加当前运行记录
            run_record = {
                metric_type: current_metric,
                "epoch": epoch,
                "seed": seed,
                "metrics": self._convert_tensors_to_python(metrics),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "metric_type": metric_type
            }
            self.best_hos_history["all_runs"].append(run_record)

            # 只保留最近10次运行记录
            if len(self.best_hos_history["all_runs"]) > 10:
                self.best_hos_history["all_runs"] = self.best_hos_history["all_runs"][-10:]

            # 保存历史记录到文件
            self.save_history_best()

            if type(optimizers) not in [list, tuple]:
                optimizers = [optimizers]

            state = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizers": [opt.state_dict() for opt in optimizers],
                "best_hos_history": self.best_hos_history,
                "current_run_best": self.current_run_best,
                "metrics": metrics
            }
            path = os.path.join(self.ckpt_dir, "best.pth")
            torch.save(state, path)
            print(f"🏆 历史最佳模型已保存! {metric_type.upper()}: {current_metric:.4f} (历史最佳)")

            # 打印保存的指标
            print("📊 历史最佳模型指标:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)) or (isinstance(value, torch.Tensor) and value.numel() == 1):
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    print(f"   {key}: {value:.4f}")
                elif isinstance(value, list) or (isinstance(value, torch.Tensor) and value.numel() > 1):
                    if isinstance(value, torch.Tensor):
                        value = value.tolist()
                    print(f"   {key}: {[f'{v:.4f}' for v in value]}")
                else:
                    print(f"   {key}: {value}")
        else:
            print(
                f"📊 当前{metric_type.upper()}: {current_metric:.4f}, 历史最佳: {self.best_hos_history['best_hos']:.4f}, 当前运行最佳: {self.current_run_best['best_hos']:.4f}")

    def train(self, hook: str, dataloader: Loader, epochs: int, resume: bool = False, seed: int = None):
        self.model.train()
        progress = ProgressLogger(epochs)
        self.model.progress = progress

        optimizer = getattr(self.model, f'{hook}_optimizer')()
        if type(optimizer) not in [list, tuple]:
            optimizer = [optimizer]
        loop_step: Callable = getattr(self.model, f'{hook}_step')

        # 安全地获取 epoch_end 函数，如果不存在则返回空字典的函数
        epoch_end_func = getattr(self.model, f'{hook}_epoch_end', lambda: {})
        if not callable(epoch_end_func):
            epoch_end_func = lambda: {}

        start_epoch = 0
        if resume:
            start_epoch = self.load_checkpoint(optimizers=optimizer)  # 恢复训练

        # 初始化当前运行记录（如果是新运行）
        if not resume:
            self.current_run_best = {
                "best_hos": 0.0,
                "best_epoch": 0,
                "current_seed": seed,
                "best_metrics": {},
                "run_start_time": time.time()
            }
            self.save_current_run_best()

        for epoch in range(start_epoch, epochs):
            for data in dataloader:
                data = dataToDevice(data, self.device)
                step_out = loop_step(data)
                loss, information = parseTrainStepOut(step_out)

                with OptimizerManager(optimizer):
                    loss.backward()

                progress.add_information(information)

            # 调用 epoch_end 函数并处理可能的异常
            try:
                epoch_out = epoch_end_func()
            except Exception as e:
                print(f"⚠️ Error in {hook}_epoch_end: {e}")
                epoch_out = {}

            # 确保 epoch_out 是一个字典
            if epoch_out is None:
                epoch_out = {}

            progress.update(epoch_out)

            # 只有在 epoch_out 是字典时才进行处理
            if isinstance(epoch_out, dict):
                # 训练阶段不要求完整评估指标集
                self.check_required_metrics(epoch_out, 'train')

                # 优先使用 HOS 作为评判标准（如果存在）
                if 'hos' in epoch_out:
                    # 确保hos值是Python原生类型
                    hos_value = epoch_out['hos']
                    if isinstance(hos_value, torch.Tensor):
                        hos_value = hos_value.item()
                    self.save_best_checkpoint(epoch + 1, optimizer, hos_value, epoch_out, seed, "hos")
                # 其次使用 source_oa 作为评判标准（如果存在）
                elif 'source_oa' in epoch_out:
                    source_oa_value = epoch_out['source_oa']
                    if isinstance(source_oa_value, torch.Tensor):
                        source_oa_value = source_oa_value.item()
                    self.save_best_checkpoint(epoch + 1, optimizer, source_oa_value, epoch_out, seed, "source_oa")
                # 再次使用 accuracy 或其他可能的指标
                elif 'accuracy' in epoch_out:
                    accuracy = epoch_out['accuracy']
                    if isinstance(accuracy, torch.Tensor):
                        accuracy = accuracy.item()
                    self.save_best_checkpoint(epoch + 1, optimizer, accuracy, epoch_out, seed, "accuracy")
                # 如果没有找到任何评估指标，则提示警告
                else:
                    print("⚠️ 警告: 训练输出中没有找到任何可用的评估指标(hos/source_oa/accuracy)")

            # 每个epoch保存一次last.pth
            self.save_checkpoint(epoch + 1, optimizer, name="last.pth", metrics=epoch_out)

        progress.close()

    def test(self, hook: str, dataloader: Loader, seed: int = None):
        self.model.eval()

        loop_step: Callable = getattr(self.model, f'{hook}_step')
        test_end = getFunc(self.model, f'{hook}_end')

        with torch.no_grad():
            for data in tqdm(dataloader):
                data = dataToDevice(data, self.device)
                loop_step(data)

            test_result = test_end()

            # 如果是测试阶段，也保存最佳模型（基于测试集的指标）
            if hook == 'test' and test_result is not None and isinstance(test_result, dict) and test_result:
                # 获取优化器用于保存checkpoint
                try:
                    optimizer = getattr(self.model, 'train_optimizer')()
                    if type(optimizer) not in [list, tuple]:
                        optimizer = [optimizer]

                    # 在测试阶段，检查完整的评估指标集
                    has_required_metrics = self.check_required_metrics(test_result, 'test')
                    if not has_required_metrics:
                        print("⚠️ 测试结果缺少部分必需指标，但仍会尝试保存检查点")

                    # 优先使用 HOS 值作为评判标准
                    if 'hos' in test_result:
                        # 确保hos值是Python原生类型
                        hos_value = test_result['hos']
                        if isinstance(hos_value, torch.Tensor):
                            hos_value = hos_value.item()
                        self.save_best_checkpoint(0, optimizer, hos_value, test_result, seed, "hos")
                        print(f"📊 测试结果 HOS: {hos_value:.4f}")
                    # 如果没有HOS，但有OA，则使用OA
                    elif 'oa' in test_result:
                        oa_value = test_result['oa']
                        if isinstance(oa_value, torch.Tensor):
                            oa_value = oa_value.item()
                        self.save_best_checkpoint(0, optimizer, oa_value, test_result, seed, "oa")
                        print(f"📊 测试结果 OA: {oa_value:.4f}")
                    else:
                        print("⚠️ 警告: 测试结果中缺少HOS和OA值，无法保存最佳模型")

                    # 打印所有测试指标
                    print("\n📊 测试结果详情:")
                    for key, value in test_result.items():
                        if isinstance(value, (int, float)) or (isinstance(value, torch.Tensor) and value.numel() == 1):
                            if isinstance(value, torch.Tensor):
                                value = value.item()
                            print(f"   {key}: {value:.4f}")
                        elif isinstance(value, list) or (isinstance(value, torch.Tensor) and value.numel() > 1):
                            if isinstance(value, torch.Tensor):
                                value = value.tolist()
                            if len(value) > 10:  # 如果列表太长，只显示前几个
                                print(f"   {key}: [{', '.join([f'{v:.4f}' for v in value[:5]])}...(共{len(value)}项)]")
                            else:
                                print(f"   {key}: [{', '.join([f'{v:.4f}' for v in value])}]")
                        else:
                            print(f"   {key}: {value}")

                except Exception as e:
                    print(f"⚠️ Could not save best model after test: {e}")
                    import traceback
                    traceback.print_exc()
            elif hook == 'test':
                print("⚠️ Test results are empty or invalid, skipping best model save")

            return test_result

    def print_history_summary(self):
        """打印历史最佳记录摘要"""
        print(f"\n{'=' * 60}")
        print("📈 历史最佳记录摘要")
        print(f"{'=' * 60}")
        metric_type = self.best_hos_history.get("metric_type", "hos")
        print(f"历史最佳{metric_type.upper()}: {self.best_hos_history['best_hos']:.4f}")
        print(f"最佳epoch: {self.best_hos_history['best_epoch']}")
        print(f"最佳seed: {self.best_hos_history['best_seed']}")
        print(f"总运行次数: {len(self.best_hos_history['all_runs'])}")

        if self.best_hos_history['all_runs']:
            print(f"\n最近运行记录:")
            for i, run in enumerate(self.best_hos_history['all_runs'][-5:]):  # 只显示最近5次
                metric_type = run.get("metric_type", "hos")
                metric_value = run.get(metric_type, 0)
                print(
                    f"  运行 {i + 1}: {metric_type.upper()}={metric_value:.4f}, seed={run['seed']}, epoch={run['epoch']}")

        print(f"\n📊 当前运行最佳记录:")
        current_metric_type = self.current_run_best.get("metric_type", "hos")
        print(f"当前运行最佳{current_metric_type.upper()}: {self.current_run_best['best_hos']:.4f}")
        print(f"当前运行最佳epoch: {self.current_run_best['best_epoch']}")
        print(f"当前运行seed: {self.current_run_best['current_seed']}")

        # 计算运行时间
        if 'run_start_time' in self.current_run_best:
            run_time = time.time() - self.current_run_best['run_start_time']
            print(f"当前运行时间: {run_time:.2f}秒")

        print(f"{'=' * 60}")


def parseTrainStepOut(step_out: Collecter) -> Sequence:
    out_type = type(step_out)

    if out_type == dict:
        loss = step_out['loss']
        information = step_out['information']
    elif out_type == list or out_type == tuple:
        loss = step_out[0]
        information = step_out[1]
    else:
        loss = step_out
        information = dict(loss=loss)

    return loss, information