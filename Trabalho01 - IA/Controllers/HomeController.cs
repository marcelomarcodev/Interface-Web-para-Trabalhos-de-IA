using System.Diagnostics;
using Microsoft.AspNetCore.Mvc;
using Trabalho01___IA.Models;

namespace Trabalho01___IA.Controllers;

public class HomeController : Controller
{
    private static readonly int[][] BipolarInputs =
    [
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1]
    ];

    private static readonly int[][] LetterAMatrix =
    [
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1]
    ];

    private static readonly int[][] LetterBMatrix =
    [
        [1, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 0, 0]
    ];

    public IActionResult Index()
    {
        return View();
    }

    public IActionResult Trabalho1()
    {
        var gateResults = new List<HebbGateResult>();

        foreach (var gate in LogicGateDefinition.GetDefaultGates())
        {
            float w1 = 0;
            float w2 = 0;
            float bias = 0;

            for (var i = 0; i < BipolarInputs.Length; i++)
            {
                var x1 = BipolarInputs[i][0];
                var x2 = BipolarInputs[i][1];
                var y = gate.Targets[i];

                w1 += x1 * y;
                w2 += x2 * y;
                bias += y;
            }

            var samples = new List<HebbSampleResult>();

            for (var i = 0; i < BipolarInputs.Length; i++)
            {
                var sum = (BipolarInputs[i][0] * w1) + (BipolarInputs[i][1] * w2) + bias;
                var prediction = sum >= 0 ? 1 : -1;

                samples.Add(new HebbSampleResult
                {
                    X1 = BipolarInputs[i][0],
                    X2 = BipolarInputs[i][1],
                    Target = gate.Targets[i],
                    Prediction = prediction
                });
            }

            gateResults.Add(new HebbGateResult
            {
                GateName = gate.Name,
                W1 = w1,
                W2 = w2,
                Bias = bias,
                Samples = samples
            });
        }

        return View(new Work1ViewModel { GateResults = gateResults });
    }

    [HttpGet]
    public IActionResult Trabalho2()
    {
        var model = BuildDefaultWork2Model();
        return View(model);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    public IActionResult Trabalho2(Work2ViewModel model)
    {
        PopulateWork2StaticData(model, model.SelectedGate);

        if (!ModelState.IsValid)
        {
            return View(model);
        }

        var gate = GetGateByName(model.SelectedGate);

        if (gate is null)
        {
            ModelState.AddModelError(nameof(model.SelectedGate), "Porta lógica inválida.");
            return View(model);
        }

        var sampleVectors = BuildLetterPairs();
        var featureLength = sampleVectors[0].Vector.Length;
        var weights = new float[featureLength];
        var bias = 0f;
        var converged = false;
        var epochsUsed = 0;

        for (var epoch = 1; epoch <= model.MaxEpochs; epoch++)
        {
            var errors = 0;

            for (var i = 0; i < sampleVectors.Count; i++)
            {
                var output = Predict(sampleVectors[i].Vector, weights, bias);
                var error = gate.Targets[i] - output;

                if (error == 0)
                {
                    continue;
                }

                for (var j = 0; j < featureLength; j++)
                {
                    weights[j] += model.LearningRate * error * sampleVectors[i].Vector[j];
                }

                bias += model.LearningRate * error;
                errors++;
            }

            epochsUsed = epoch;

            if (errors != 0)
            {
                continue;
            }

            converged = true;
            break;
        }

        var sampleResults = new List<PerceptronSampleResult>();

        for (var i = 0; i < sampleVectors.Count; i++)
        {
            sampleResults.Add(new PerceptronSampleResult
            {
                PairName = sampleVectors[i].Name,
                Target = gate.Targets[i],
                Prediction = Predict(sampleVectors[i].Vector, weights, bias)
            });
        }

        model.HasResult = true;
        model.Converged = converged;
        model.EpochsUsed = epochsUsed;
        model.Bias = bias;
        model.Weights = weights;
        model.Samples = sampleResults;

        return View(model);
    }

    public IActionResult Privacy()
    {
        return View();
    }

    [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
    public IActionResult Error()
    {
        return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
    }

    private static Work2ViewModel BuildDefaultWork2Model()
    {
        var model = new Work2ViewModel();
        PopulateWork2StaticData(model, model.SelectedGate);
        return model;
    }

    private static void PopulateWork2StaticData(Work2ViewModel model, string selectedGate)
    {
        var gate = GetGateByName(selectedGate) ?? LogicGateDefinition.GetDefaultGates()[0];

        model.GateNames = LogicGateDefinition.GetDefaultGates().Select(item => item.Name).ToList();
        model.MatrixA = LetterAMatrix.Select(row => row.ToArray()).ToList();
        model.MatrixB = LetterBMatrix.Select(row => row.ToArray()).ToList();
        model.TruthTable =
        [
            new TruthTableRow { X1 = 1, X2 = 1, Target = gate.Targets[0] },
            new TruthTableRow { X1 = 1, X2 = -1, Target = gate.Targets[1] },
            new TruthTableRow { X1 = -1, X2 = 1, Target = gate.Targets[2] },
            new TruthTableRow { X1 = -1, X2 = -1, Target = gate.Targets[3] }
        ];
    }

    private static LogicGateDefinition? GetGateByName(string gateName)
    {
        return LogicGateDefinition.GetDefaultGates()
            .FirstOrDefault(item => item.Name.Equals(gateName, StringComparison.OrdinalIgnoreCase));
    }

    private static int Predict(IReadOnlyList<int> input, IReadOnlyList<float> weights, float bias)
    {
        var sum = bias;

        for (var i = 0; i < input.Count; i++)
        {
            sum += input[i] * weights[i];
        }

        return sum >= 0 ? 1 : -1;
    }

    private static List<(string Name, int[] Vector)> BuildLetterPairs()
    {
        var a = FlattenToBipolar(LetterAMatrix);
        var b = FlattenToBipolar(LetterBMatrix);

        return
        [
            ("A + A", Combine(a, a)),
            ("A + B", Combine(a, b)),
            ("B + A", Combine(b, a)),
            ("B + B", Combine(b, b))
        ];
    }

    private static int[] FlattenToBipolar(IReadOnlyList<int[]> matrix)
    {
        var result = new int[matrix.Count * matrix[0].Length];
        var index = 0;

        foreach (var row in matrix)
        {
            foreach (var value in row)
            {
                result[index++] = value == 1 ? 1 : -1;
            }
        }

        return result;
    }

    private static int[] Combine(IReadOnlyList<int> first, IReadOnlyList<int> second)
    {
        var result = new int[first.Count + second.Count];

        for (var i = 0; i < first.Count; i++)
        {
            result[i] = first[i];
        }

        for (var i = 0; i < second.Count; i++)
        {
            result[first.Count + i] = second[i];
        }

        return result;
    }
}
