using System.Collections.Generic;
namespace FeaServer.Engine.Time.Scheduler
{
    internal struct Hibernate
    {
        public ElementCollection Elements;

        public Hibernate(int none)
        {
            Elements = new ElementCollection(0);
        }
    }
}
