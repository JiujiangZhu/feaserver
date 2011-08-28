using System.Collections.Generic;
using System.Runtime.InteropServices;
using System;
namespace FeaServer.Engine.Time.Scheduler
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct HibernateCollection
    {
        private Hibernate[] _hibernates;

        public HibernateCollection(int none)
        {
            _hibernates = new Hibernate[EngineSettings.MaxHibernates];
            for (int hibernateIndex = 0; hibernateIndex < _hibernates.Length; hibernateIndex++)
                _hibernates[hibernateIndex] = new Hibernate(0);
        }

        public void Hibernate(Element element, ulong time)
        {
            Console.WriteLine("Timeline: Hibernate %d", TimePrecision.DecodeTime(time));
            var hibernate = _hibernates[0];
            hibernate.Elements.Add(element, BitConverter.GetBytes(time));
        }

        public void DeHibernate(Action<Element, ulong> addAction)
        {
            var hibernate = _hibernates[0];
            hibernate.Elements.WalkAndRemove((element, metadata) =>
            {
                var time = BitConverter.ToUInt64(metadata, 0);
                if (time < EngineSettings.MaxTimeslicesTime)
                    throw new Exception("paranoia");
                var newTime = (ulong)(time -= EngineSettings.MaxTimeslicesTime);
                return (newTime < EngineSettings.MaxTimeslicesTime ? (object)newTime : null);
            }, (element, metadata, newTime) =>
            {
                Console.WriteLine("Timeline: Dehibernate {%d}", TimePrecision.DecodeTime((ulong)newTime));
                // add to timeline
                addAction(element, (ulong)newTime);
            });
        }
    }
}
