import os
import subprocess
import json
import glob
from datetime import datetime

# UEA 数据集列表
UEA_DATASETS = [
    #'ArticularyWordRecognition',
    #'BasicMotions',
    #'CharacterTrajectories',
    #'Cricket',
    #'DuckDuckGeese',# 慢
    #'ERing',
    #'EigenWorms',
    #'Epilepsy',
    #'EthanolConcentration',
    #'FaceDetection',# 慢
    'FingerMovements',
    'HandMovementDirection',
    'Handwriting',
    'Heartbeat',
    'JapaneseVowels',
    'LSST',
    'Libras',
    'MotorImagery',
    'NATOPS',
    'PEMS-SF',
    'PenDigits',
    'PhonemeSpectra',
    'RacketSports',
    'SelfRegulationSCP1',
    'SelfRegulationSCP2',
    'SpokenArabicDigits',
    'StandWalkJump',
    'UWaveGestureLibrary',
]

def find_latest_result(dataset_name):
    """找到数据集最新的 linear_result.txt 文件"""
    pattern = f"Results/Rep-Learning/{dataset_name}/**/linear_result.txt"
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    # 按修改时间排序，取最新的
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def read_final_accuracy(result_file):
    """读取 linear_result.txt 的最后一行，获取最终准确率"""
    try:
        with open(result_file, 'r') as f:
            lines = f.readlines()
            if not lines:
                return None
            # 最后一行格式: epoch, test_acc, align_loss, std_loss, cov_loss
            last_line = lines[-1].strip()
            parts = last_line.split(',')
            if len(parts) >= 2:
                return float(parts[1])  # test_acc 是第2列
    except Exception as e:
        print(f"Warning: Failed to read accuracy from {result_file}: {e}")
    return None

def run_dataset(dataset_name):
    """运行单个数据集"""
    print(f"\n{'='*60}")
    print(f"Running dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # 简化：只传 data_dir，其他用 main.py 的默认值
    cmd = ['python', 'main.py', '--data_dir', dataset_name]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {dataset_name} completed successfully!")
        
        # 读取最终准确率
        result_file = find_latest_result(dataset_name)
        final_acc = read_final_accuracy(result_file) if result_file else None
        
        return {'success': True, 'final_acc': final_acc, 'result_file': result_file}
    except subprocess.CalledProcessError as e:
        print(f"❌ {dataset_name} failed!")
        return {'success': False, 'final_acc': None, 'result_file': None}

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    summary_file = f"uea_batch_results_{timestamp}.json"
    
    results = {'timestamp': timestamp, 'datasets': {}}
    
    for dataset in UEA_DATASETS:
        result = run_dataset(dataset)
        results['datasets'][dataset] = result
        
        # 保存中间结果
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # 打印汇总
    total = len(UEA_DATASETS)
    success_count = sum(1 for v in results['datasets'].values() if v['success'])
    print(f"\n{'='*60}")
    print(f"BATCH RUN SUMMARY: {success_count}/{total} succeeded")
    print(f"\nDetailed Results:")
    print(f"{'Dataset':<30} {'Success':<10} {'Test Acc':<10}")
    print("-" * 60)
    for dataset, info in results['datasets'].items():
        acc_str = f"{info['final_acc']:.4f}" if info['final_acc'] else "N/A"
        print(f"{dataset:<30} {str(info['success']):<10} {acc_str:<10}")
    print(f"\nResults saved to: {summary_file}")

if __name__ == '__main__':
    main()