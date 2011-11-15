
//private void LoadGroup(IEnumerable<Compound> compounds)
//{
//    Clear();
//    ulong size = 0L;
//    foreach (var compound in compounds)
//        size += PrepareCompound(compound);
//    var offset = XMalloc(size);
//    XTransfer();
//    // INIT|Every
//}

//private ulong PrepareCompound(Compound compound)
//{
//    var w = MainW;
//    var spec = CompoundSpec<TEngine>.GetSpec(compound.Type);
//    var specW = GetSpecWriter(spec);
//    int specLength = spec.Length;
//    MainW.Write(LOADSTORE_MAGIC);
//    w.Write(specLength);
//    //var size = (ulong)((specLength * 8L) + 8L);
//    var size = array_getSize(SizeOfSize_t, specLength);
//    var typesSizeInBytes = spec.TypesSizeInBytes;
//    for (int index = 0; index < specLength; index++)
//    {
//        int pitch = typesSizeInBytes[index];
//        int itemLength = compound.Elements[index].Length;
//        size += (ulong)((pitch * itemLength) + 8L);
//        MainW.Write(pitch);
//        MainW.Write(itemLength);
//    }

//    //// write to every stream
//    //if (spec.ScheduleStyleEveryTypeIndexs != null)
//    //{
//    //    var everyW = specW.Every;
//    //    foreach (int typeIndex in spec.ScheduleStyleEveryTypeIndexs)
//    //    {
//    //        everyW.Write(0);
//    //    }
//    //}

//    //int[] typesSizeInBytes = null;
//    //byte[] data = null;
//    //int n2 = 0;
//    //int dataSize = 0;
//    //typesSizeInBytes = spec.TypesSizeInBytes;
//    //for (int index = 0; index < spec.Length; index++)
//    //{
//    //    byte num;
//    //    int pitch = typesSizeInBytes[index];
//    //    size += (ulong)((pitch * n2) + 8L);
//    //    w.Write(pitch);
//    //    w.Write(n2);
//    //    var specW = GetSpecWriter(spec);
//    //    var specDataW = specW.Data;
//    //    specDataW.Write(dataSize);
//    //    specDataW.Write(data, 0, dataSize);
//    //}
//    return size;
//}