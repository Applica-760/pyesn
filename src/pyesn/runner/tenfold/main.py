from . import utils
from . import preparation
from . import execution

def run_tenfold(cfg, *, parallel: bool = True): # 変更点: max_workers引数を削除
    # 1. 実行環境の準備
    env = preparation.prepare_run_environment(cfg)
    hp_combos = utils.flatten_search_space(getattr(cfg.train.tenfold, "search_space", None))
    
    # 変更点: YAMLからworkersの値を取得。デフォルトはos.cpu_count() or 1
    max_workers = getattr(cfg.train.tenfold, "workers", None)
    print(max_workers)
    if max_workers is None:
        import os
        max_workers = os.cpu_count() or 1
        print(f"[WARN] 'workers' not found in config. Defaulting to {max_workers}.")


    # 変更点: 全てのタスクを事前にフラットなリストとして生成する
    print("[INFO] Generating a master list of all tasks to run...")
    all_tasks = []
    for hp_overrides, hp_tag in hp_combos:
        # 実行すべきタスク（fold）を決定
        tasks_to_run_for_hp = preparation.determine_tasks_to_run(
            cfg, hp_overrides, env["letters"], env["weight_dir"]
        )

        # 各タスクに必要な情報を辞書として格納
        for i, leave_letter in enumerate(tasks_to_run_for_hp):
            task_info = {
                "hp_overrides": hp_overrides,
                "hp_tag": hp_tag,
                "leave_out_letter": leave_letter,
                "seed": i  # シード値もここで決定
            }
            all_tasks.append(task_info)

    if not all_tasks:
        print("[INFO] All tasks are already completed. Nothing to do.")
    else:
        print(f"[INFO] Total {len(all_tasks)} tasks will be executed.")
        # 変更点: フラット化されたタスクリストを一括で実行関数に渡す
        execution.execute_tasks(
            cfg, env, all_tasks, parallel, max_workers
        )

    print("="*50)
    print("[INFO] 10-fold hyperparameter search finished.")
    print("="*50)