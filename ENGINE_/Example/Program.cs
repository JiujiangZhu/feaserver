using System;
using FeaServer.Engine;
using Example.Mocks;
namespace Example
{
    public class Program
    {
        static void Main(string[] args)
        {
            var elementTypes = new[] { new MockElementType() };
            using (var engine = EngineFactory.Create(elementTypes, EngineProvider.Managed))
            {

            }
        }
    }
}
