namespace Trabalho01___IA.Models;

public sealed class LogicGateDefinition
{
    public required string Name { get; init; }
    public required int[] Targets { get; init; }

    public static IReadOnlyList<LogicGateDefinition> GetDefaultGates()
    {
        return
        [
            new LogicGateDefinition { Name = "AND", Targets = [1, -1, -1, -1] },
            new LogicGateDefinition { Name = "OR", Targets = [1, 1, 1, -1] },
            new LogicGateDefinition { Name = "NAND", Targets = [-1, 1, 1, 1] },
            new LogicGateDefinition { Name = "NOR", Targets = [-1, -1, -1, 1] },
            new LogicGateDefinition { Name = "XOR", Targets = [-1, 1, 1, -1] }
        ];
    }
}
