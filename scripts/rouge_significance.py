import argparse
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from rouge import Rouge
import os
import tempfile


class PairedBootstrapOutput(NamedTuple):
    baseline_bleu: float
    new_bleu: float
    num_samples: int
    baseline_better: int
    num_equal: int
    new_better: int
    p_value: float


def get_sentence_level_stats(
    translations: List[str], references: List[str], metric: str
) -> pd.DataFrame:
    """
    方式1：计算每个句子的ROUGE分数
    """
    assert len(translations) == len(references), (
        f"There are {len(translations)} translated sentences "
        f"but {len(references)} reference sentences"
    )
    
    rouge = Rouge()
    sufficient_stats: List[List[float]] = []
    
    results = rouge.get_scores(translations, references, avg=False)
    
    for r in results:
        sufficient_stats.append([r[metric]["f"]])
    
    return pd.DataFrame(sufficient_stats, columns=["fscore"])

def calculate_system_level_rouge_python(predictions, references, metric='rouge-l'):
    """
    纯Python实现的系统级ROUGE（不依赖Perl）
    将所有句子合并成一个文档后计算
    """
    from rouge import Rouge
    rouge = Rouge()
    
    # 合并所有句子成一个大文档
    combined_predictions = ' '.join(predictions)
    combined_references = ' '.join(references)
    
    scores = rouge.get_scores(combined_predictions, combined_references)
    return scores[0][metric]['f']

def calculate_system_level_rouge(predictions, references, metric='rouge-l', temp_dir="/temprouge"):
    """
    使用files2rouge包计算系统级ROUGE
    
    Args:
        predictions: 预测文本列表
        references: 参考文本列表
        metric: ROUGE指标类型
        temp_dir: 临时文件保存目录，None则使用系统默认
    """
    # 方法1：指定临时文件目录
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(
        mode='w', 
        suffix='.txt', 
        delete=False,  # 设为False以便手动控制删除
        dir=temp_dir   # 指定目录
    ) as pred_file:
        pred_file.write('\n'.join(predictions))
        pred_path = pred_file.name
    
    with tempfile.NamedTemporaryFile(
        mode='w', 
        suffix='.txt', 
        delete=False,
        dir=temp_dir
    ) as ref_file:
        ref_file.write('\n'.join(references))
        ref_path = ref_file.name
    
    try:
        # 调用files2rouge
        scores = files2rouge.run(pred_path, ref_path)
        return scores[metric]['f']
    finally:
        # 清理临时文件（确保删除）
        if os.path.exists(pred_path):
            os.unlink(pred_path)
        if os.path.exists(ref_path):
            os.unlink(ref_path)


def calculate_rouge_for_sample(
    translations: List[str], 
    references: List[str], 
    indices: np.ndarray, 
    metric: str
) -> float:
    """
    方式2：计算采样句子的整体ROUGE分数
    """
    rouge = Rouge()
    
    sampled_translations = [translations[i] for i in indices]
    sampled_references = [references[i] for i in indices]
    
    try:
        scores = rouge.get_scores(sampled_translations, sampled_references, avg=True)
        return scores[metric]["f"]
    except:
        return 0.0


def method1_bootstrap_resample(
    baseline_stats: pd.DataFrame,
    new_stats: pd.DataFrame,
    num_samples: int = 1000,
    sample_size: Optional[int] = None
) -> PairedBootstrapOutput:
    """
    方式1：对已计算的句子级ROUGE分数进行bootstrap
    """
    assert len(baseline_stats) == len(new_stats), (
        f"Length mismatch - baseline has {len(baseline_stats)} lines "
        f"while new has {len(new_stats)} lines."
    )
    
    num_sentences = len(baseline_stats)
    if not sample_size:
        sample_size = num_sentences
    
    # t-test for paired samples
    ttest_results = stats.ttest_rel(baseline_stats.fscore, new_stats.fscore)
    
    indices = np.random.randint(
        low=0, high=num_sentences, size=(num_samples, sample_size)
    )
    
    baseline_better = 0
    new_better = 0
    num_equal = 0
    score_differences = []
    
    for index in indices:
        baseline_bleu = baseline_stats.iloc[index].mean().item()
        new_bleu = new_stats.iloc[index].mean().item()
        
        if new_bleu > baseline_bleu:
            new_better += 1
        elif baseline_bleu > new_bleu:
            baseline_better += 1
        else:
            num_equal += 1
        
        score_differences.append(new_bleu - baseline_bleu)
    
    # 计算p-value
    score_differences = np.array(score_differences)
    p_value = np.sum(score_differences <= 0) / num_samples
    
    return PairedBootstrapOutput(
        baseline_bleu=baseline_stats.mean().item(),
        new_bleu=new_stats.mean().item(),
        num_samples=num_samples,
        baseline_better=baseline_better,
        num_equal=num_equal,
        new_better=new_better,
        p_value=p_value
    ), ttest_results


def method2_bootstrap_resample(
    baseline_translations: List[str],
    new_translations: List[str],
    references: List[str],
    num_samples: int = 1000,
    sample_size: Optional[int] = None,
    metric: str = 'rouge-l'
) -> PairedBootstrapOutput:
    """
    方式2：先采样，再计算整体ROUGE的bootstrap方法
    """
    assert len(baseline_translations) == len(new_translations) == len(references), (
        f"Length mismatch - baseline: {len(baseline_translations)}, "
        f"new: {len(new_translations)}, references: {len(references)}"
    )
    
    num_sentences = len(baseline_translations)
    if not sample_size:
        sample_size = num_sentences
    
    # 计算原始数据的ROUGE分数（使用files2rouge）
    original_baseline_score = calculate_system_level_rouge(
        baseline_translations, references, metric
    )
    original_new_score = calculate_system_level_rouge(
        new_translations, references, metric
    )
    
    print(f"Original baseline {metric}: {original_baseline_score:.4f}")
    print(f"Original new {metric}: {original_new_score:.4f}")
    
    # Bootstrap采样
    baseline_better = 0
    new_better = 0
    num_equal = 0
    score_differences = []
    
    for _ in range(num_samples):
        indices = np.random.randint(low=0, high=num_sentences, size=sample_size)
        
        # 修复：正确采样句子
        sampled_baseline = [baseline_translations[i] for i in indices]
        sampled_new = [new_translations[i] for i in indices]
        sampled_refs = [references[i] for i in indices]

        # 计算采样后的系统级ROUGE
        baseline_score = calculate_system_level_rouge(
            sampled_baseline, sampled_refs, metric  # 修复：传递正确的参数
        )
        new_score = calculate_system_level_rouge(
            sampled_new, sampled_refs, metric  # 修复：传递正确的参数
        )
        
        if new_score > baseline_score:
            new_better += 1
        elif baseline_score > new_score:
            baseline_better += 1
        else:
            num_equal += 1
        
        score_differences.append(new_score - baseline_score)
    
    # 计算p-value
    score_differences = np.array(score_differences)
    p_value = np.sum(score_differences <= 0) / num_samples
    
    return PairedBootstrapOutput(
        baseline_bleu=original_baseline_score,
        new_bleu=original_new_score,
        num_samples=num_samples,
        baseline_better=baseline_better,
        num_equal=num_equal,
        new_better=new_better,
        p_value=p_value
    )


def load_sentences(file):
    data = []
    with open(file) as f:
        for line in f.readlines():
            data.append(line.strip("\n"))
    return data


def paired_bootstrap_resample_from_files(
    reference_file: str,
    baseline_files: list,
    new_files: list,
    num_samples: int = 1000,
    sample_size: Optional[int] = None,
    metric: str = 'rouge-l',
    opt_method2: bool = False
) -> Tuple[PairedBootstrapOutput, PairedBootstrapOutput]:
    """
    执行两种bootstrap方法
    """
    # 加载数据
    references_single = load_sentences(reference_file)
    
    baseline_translations = []
    references = []
    for baseline_file in baseline_files:
        baseline_translations += load_sentences(baseline_file)
        references += references_single
    
    new_translations = []
    for new_file in new_files:
        new_translations += load_sentences(new_file)
    
    print(f"Loaded {len(baseline_translations)} sentences for comparison")
    print(f"Number of bootstrap samples: {num_samples}")
    print(f"Metric: {metric}\n")
    
    # 方式1：句子级ROUGE后bootstrap
    print("="*60)
    print("Method 1: Sentence-level ROUGE then Bootstrap")
    print("="*60)
    
    baseline_stats = get_sentence_level_stats(
        translations=baseline_translations, references=references, metric=metric
    )
    new_stats = get_sentence_level_stats(
        translations=new_translations, references=references, metric=metric
    )
    
    method1_output, ttest_results = method1_bootstrap_resample(
        baseline_stats=baseline_stats,
        new_stats=new_stats,
        num_samples=num_samples,
        sample_size=sample_size
    )
    
    print(f"T-test results: t-statistic={ttest_results.statistic:.4f}, p-value={ttest_results.pvalue:.4f}")
    
    # 打印结果
    print_results(method1_output, "Method 1 (Sentence-level ROUGE)")

    method2_output = None
    if opt_method2:
        from files2rouge import files2rouge

        # 方式2：bootstrap采样后计算整体ROUGE
        print("\n" + "="*60)
        print("Method 2: Bootstrap Sampling then System-level ROUGE")
        print("="*60)
        
        method2_output = method2_bootstrap_resample(
            baseline_translations=baseline_translations,
            new_translations=new_translations,
            references=references,
            num_samples=num_samples,
            sample_size=sample_size,
            metric=metric
        )
        print_results(method2_output, "Method 2 (System-level ROUGE)")
    
    return method1_output, method2_output


def print_results(output: PairedBootstrapOutput, method_name: str):
    """
    打印bootstrap结果
    """
    print(f"\n{method_name} Results:")
    print("-"*40)
    print(f"Baseline F1-score: {output.baseline_bleu:.4f}")
    print(f"New F1-score: {output.new_bleu:.4f}")
    print(f"F1-score delta: {output.new_bleu - output.baseline_bleu:.4f}")
    print(f"\nBootstrap Statistics ({output.num_samples} samples):")
    print(f"  Baseline better: {output.baseline_better} ({output.baseline_better / output.num_samples:.1%})")
    print(f"  New better: {output.new_better} ({output.new_better / output.num_samples:.1%})")
    print(f"  Equal: {output.num_equal} ({output.num_equal / output.num_samples:.1%})")
    print(f"  p-value: {output.p_value:.4f}")
    
    if output.p_value < 0.001:
        significance = "*** (p < 0.001)"
    elif output.p_value < 0.01:
        significance = "** (p < 0.01)"
    elif output.p_value < 0.05:
        significance = "* (p < 0.05)"
    elif output.p_value < 0.1:
        significance = "(p < 0.1, marginal)"
    else:
        significance = "(not significant)"
    
    print(f"  Significance: {significance}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-file",
        type=str,
        required=True,
        help="Text file containing reference sentences.",
    )
    parser.add_argument(
        "--baseline-files",
        nargs='+',
        required=True,
        help="Text files containing sentences from baseline system.",
    )
    parser.add_argument(
        "--new-files",
        nargs='+',
        required=True,
        help="Text files containing sentences from new system.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="rouge-l",
        choices=['rouge-1', 'rouge-2', 'rouge-l'],
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of bootstrap samples (default: 1000)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Size of each bootstrap sample (default: same as data size)",
    )
    
    args = parser.parse_args()
    
    # 执行两种bootstrap方法
    method1_output, method2_output = paired_bootstrap_resample_from_files(
        reference_file=args.reference_file,
        baseline_files=args.baseline_files,
        new_files=args.new_files,
        num_samples=args.num_samples,
        sample_size=args.sample_size,
        metric=args.metric,
        opt_method2=True
    )
    

if __name__ == "__main__":
    main()