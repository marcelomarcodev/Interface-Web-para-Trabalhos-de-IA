namespace Trabalho01___IA.Models;

public sealed class HebbSampleResult
{
    public int X1 { get; init; }
    public int X2 { get; init; }
    public int Target { get; init; }
    public int Prediction { get; init; }
    public bool IsCorrect => Target == Prediction;
}

public sealed class HebbGateResult
{
    public required string GateName { get; init; }
    public float W1 { get; init; }
    public float W2 { get; init; }
    public float Bias { get; init; }
    public required IReadOnlyList<HebbSampleResult> Samples { get; init; }
    public int CorrectCount => Samples.Count(sample => sample.IsCorrect);
    public bool SolvedLinearly => CorrectCount == Samples.Count;
}

public sealed class Work1ViewModel
{
    public required IReadOnlyList<HebbGateResult> GateResults { get; init; }
}
