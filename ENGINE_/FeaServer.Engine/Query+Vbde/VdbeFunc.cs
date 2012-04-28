namespace FeaServer.Engine.Query
{
    public class VdbeFunc : FuncDef
    {
        public class AuxData
        {
            public object pAux; // Aux data for the i-th argument
        }

        public FuncDef pFunc;                       // The definition of the function
        public int nAux;                            // Number of entries allocated for apAux[]
        public AuxData[] apAux = new AuxData[2];    // One slot for each function argument
    }
}