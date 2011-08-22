using System;
using FeaServer.Engine;
namespace Example
{
    class Program
    {
        static void Main(string[] args)
        {
            using (var engine = EngineFactory.Create(null, EngineProvider.CUDA))
            {
                //engine.ElementTypes
            }
        }
    }
}
