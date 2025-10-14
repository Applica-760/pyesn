import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from pathlib import Path

from . import task
from .setup import init_global_worker_env

def _append_result_to_csv(result: dict, weight_dir: Path):
    """
    1件の実行結果をDataFrameに変換し、CSVファイルに追記保存する。
    この関数はアトミックではないが、親プロセスからシーケンシャルに呼ばれるためロックは不要。
    """
    df = pd.DataFrame([result])
    df = df[["timestamp", "hp_tag", "fold", "execution_time_sec"]] # カラム順を整理
    
    csv_path = weight_dir / "execution_times.csv"
    
    file_exists = csv_path.exists()
    df.to_csv(csv_path, mode='a', header=not file_exists, index=False, float_format='%.4f')
    print(f"[INFO] Appended execution record to {csv_path.name} for fold '{result['fold']}' (HP: {result['hp_tag']})")

# 変更点: 引数をall_tasksに変更
def execute_tasks(cfg, env, all_tasks, parallel, max_workers):
    """
    タスクリストに基づき、逐次または並列で学習を実行する。
    fold完了ごとに実行結果をCSVに追記する。
    """
    if not parallel:
        _execute_sequentially(cfg, env, all_tasks)
    else:
        _execute_in_parallel(cfg, env, all_tasks, max_workers)

# 変更点: 引数をall_tasksに変更
def _execute_sequentially(cfg, env, all_tasks):
    """タスクを逐次実行し、完了ごとに結果を書き込む。"""
    print(f"[INFO] Running {len(all_tasks)} tasks sequentially.")
    for task_info in all_tasks:
        try:
            execution_time, timestamp = task.run_one_fold_search(
                cfg,
                csv_map=env["csv_map"],
                all_letters=env["letters"],
                leave_out_letter=task_info["leave_out_letter"],
                hp_overrides=task_info["hp_overrides"],
                weight_dir=env["weight_dir"],
                seed=task_info["seed"]
            )
            result = {
                "timestamp": timestamp,
                "hp_tag": task_info["hp_tag"],
                "fold": task_info["leave_out_letter"],
                "execution_time_sec": execution_time,
            }
            _append_result_to_csv(result, env["weight_dir"])
        except Exception as e:
            print(f"[ERROR] Fold '{task_info['leave_out_letter']}' in combo '{task_info['hp_tag']}' failed: {e}")

# 変更点: 引数をall_tasksに変更し、処理をフラットなタスクに対応
def _execute_in_parallel(cfg, env, all_tasks, max_workers):
    """タスクを並列実行し、完了ごとに結果を書き込む。"""
    print(f"[INFO] Running {len(all_tasks)} tasks in parallel with up to {max_workers} workers.")
    workers = min(max_workers, (os.cpu_count() or max_workers))
    executor_kwargs = {"max_workers": workers}
    if os.name == "posix":
        executor_kwargs["mp_context"] = mp.get_context("fork")

    with ProcessPoolExecutor(**executor_kwargs, initializer=init_global_worker_env) as ex:
        # futureオブジェクトと、それに紐づくタスク情報をマッピング
        future_to_task_info = {
            ex.submit(
                task.run_one_fold_search,
                cfg,
                csv_map=env["csv_map"],
                all_letters=env["letters"],
                leave_out_letter=task_info["leave_out_letter"],
                hp_overrides=task_info["hp_overrides"],
                weight_dir=env["weight_dir"],
                seed=task_info["seed"]
            ): task_info for task_info in all_tasks
        }

        for future in as_completed(future_to_task_info):
            task_info = future_to_task_info[future]
            try:
                execution_time, timestamp = future.result()
                result = {
                    "timestamp": timestamp,
                    "hp_tag": task_info["hp_tag"],
                    "fold": task_info["leave_out_letter"],
                    "execution_time_sec": execution_time,
                }
                _append_result_to_csv(result, env["weight_dir"])
            except Exception as e:
                print(f"[ERROR] Fold '{task_info['leave_out_letter']}' in combo '{task_info['hp_tag']}' failed: {e}")