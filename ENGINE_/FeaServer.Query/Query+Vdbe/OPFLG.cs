using System.Runtime.InteropServices;
using FeaServer.Core;
namespace FeaServer.Query
{
    // Properties such as "out2" or "jump" that are specified in comments following the "case" for each opcode in the vdbe.c are encoded into bitvectors as follows:
    public enum OP
    {
        JUMP = 0x0001, /* jump:  P2 holds jmp target */
        OUT2_PRERELEASE = 0x0002, /* out2-prerelease: */
        IN1 = 0x0004, /* in1:   P1 is an input */
        IN2 = 0x0008, /* in2:   P2 is an input */
        IN3 = 0x0010, /* in3:   P3 is an input */
        OUT2 = 0x0020, /* out2:  P2 is an output */
        OUT3 = 0x0040, /* out3:  P3 is an output */
    }
}