namespace FeaServer.Query
{
    /// <summary>
    /// A smaller version of VdbeOp used for the VdbeAddOpList() function because it takes up less space. 
    /// </summary>
    public class VdbeOPSlim // [:VdbeOpList:]
    {
        public byte opcode;          /* What operation to perform */
        public sbyte p1;     /* First operand */
        public sbyte p2;     /* Second parameter (often the jump destination) */
        public sbyte p3;     /* Third parameter */
    }
}
