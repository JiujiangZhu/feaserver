using System.Collections.Generic;
using System.Runtime.InteropServices;
using System;
namespace FeaServer.Engine.Time.Scheduler
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct HibernateCollection
    {
        private Hibernate[] _hibernates;

        public HibernateCollection xtor()
        {
            _hibernates = new Hibernate[EngineSettings.MaxHibernates];
            for (int hibernateIndex = 0; hibernateIndex < _hibernates.Length; hibernateIndex++)
                _hibernates[hibernateIndex].xtor();
            return this;
        }

        public void Hibernate(Element element, ulong time)
        {
            Console.WriteLine("Timeline: Hibernate %d", TimePrec.DecodeTime(time));
            var hibernate = _hibernates[0];
            hibernate.Elements.Add(element, time);
        }

        public void DeHibernate(SliceCollection slices)
        {
            var hibernate = _hibernates[0];
            hibernate.Elements.DeHibernate(slices);
        }
    }
}
