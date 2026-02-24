using System.ComponentModel.DataAnnotations;

namespace Trabalho01___IA.Models;

public sealed class PerceptronSampleResult
{
    public required string PairName { get; init; }
    public int Target { get; init; }
    public int Prediction { get; init; }
    public bool IsCorrect => Target == Prediction;
}

public sealed class TruthTableRow
{
    public int X1 { get; init; }
    public int X2 { get; init; }
    public int Target { get; init; }
}

public sealed class Work2ViewModel
{
    [Required]
    public string SelectedGate { get; set; } = "AND";

    [Range(1, 500)]
    public int MaxEpochs { get; set; } = 30;

    [Range(0.01, 10)]
    public float LearningRate { get; set; } = 1f;

    public IReadOnlyList<string> GateNames { get; set; } = [];
    public IReadOnlyList<int[]> MatrixA { get; set; } = [];
    public IReadOnlyList<int[]> MatrixB { get; set; } = [];
    public IReadOnlyList<TruthTableRow> TruthTable { get; set; } = [];

    public bool HasResult { get; set; }
    public bool Converged { get; set; }
    public int EpochsUsed { get; set; }
    public float Bias { get; set; }
    public IReadOnlyList<float> Weights { get; set; } = [];
    public IReadOnlyList<PerceptronSampleResult> Samples { get; set; } = [];
    public int CorrectCount => Samples.Count(sample => sample.IsCorrect);
}
