#pragma once
namespace FeaServer { namespace Engine {
	struct LinkKind
	{
	public:
        unsigned long HibernateTime;
        LinkKind* NextLink;

		inline static void AddFirst(LinkKind*& chain, LinkKind* value)
		{
			value->NextLink = chain;
			chain = value;
		}

		inline static void Remove(LinkKind* last, LinkKind* value)
		{
			//last = (m_hibernateLink != value ? (last->NextLink = value->NextLink) : (m_hibernateLink = value->NextLink));
			last = (last->NextLink = value->NextLink); value->NextLink = nullptr; // paranoia
		}
	};
}}
