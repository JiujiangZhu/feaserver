namespace FeaServer.Engine.Query
{
    public struct VdbeOpList
    {
        public byte opcode;     // What operation to perform
        public int p1;          // First operand
        public int p2;          // Second parameter (often the jump destination)
        public int p3;          // Third parameter

        public VdbeOpList(byte opcode, int p1, int p2, int p3)
        {
            this.opcode = opcode;
            this.p1 = p1;
            this.p2 = p2;
            this.p3 = p3;
        }
    }
}