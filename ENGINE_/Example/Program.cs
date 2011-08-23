using System;
using FeaServer.Engine;
namespace Example
{
    public class Program
    {
        static void Main(string[] args)
        {
            var elementTypes = new[] { new SampleElementType() };
            using (var engine = EngineFactory.Create(elementTypes, EngineProvider.Cpu))
            {
            }
        }
    }
}
