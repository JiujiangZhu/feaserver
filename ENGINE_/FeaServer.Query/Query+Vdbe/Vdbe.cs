using System;
using System.IO;
using FeaServer.Query.Query;

namespace FeaServer.Query
{
    public interface IVdbe
    {
int AddOp0(int);
int AddOp1(int,int);
int AddOp2(int,int,int);
int AddOp3(int op,int p1,int p2,int p3);
int AddOp4(int op,int p1,int p2,int p3,string p4,VdbeP4Type p4Type);
int AddOp4Int(int op,int p1,int p2,int p3,int p4);
int AddOpList(int nOp, VdbeOPSlim[] aOp);
void AddParseSchemaOp(int db,string where);
void ChangeP1(uint addr, int P1);
void ChangeP2(uint addr, int P2);
void ChangeP3(uint addr, int P3);
void ChangeP5(byte P5);
void JumpHere(int addr);
void ChangeToNoop(int addr);
void ChangeP4(int addr, string zP4, int N);
void UsesBtree(int);
VdbeOP GetOp(int);
int MakeLabel();
void RunOnlyOnce();
void Delete();
void DeleteObject(Context ctx);
void MakeReady(Parse);
int Finalize();
void ResolveLabel(int x);
int CurrentAddr();
  int AssertMayAbort(int);
  void Trace(StreamWriter trace);
void ResetStepResult();
void Rewind();
int Reset();
void SetNumCols(int);
int SetColName(int, int, string, Action<object>);
void CountChanges();
Context Db();
void SetSql(string z, int n, int);
VdbeOP TakeOpArray(int[], int[]);
Value GetValue(int, byte);
void SetVarmask(int);
  string ExpandSql(string);
//void RecordUnpack(KeyInfo,int,object,UnpackedRecord);
//int RecordCompare(int,object,UnpackedRecord);
//UnpackedRecord AllocUnpackedRecord(KeyInfo , string, int, string[]);
void LinkSubProgram(SubProgram );

  void Comment(string format, params object[] args);
  void NoopComment(string format, params object[] args);

    }
        public partial class Vbde : IVdbe
    {
    }
}