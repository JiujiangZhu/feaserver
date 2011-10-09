#pragma region License
/*
The MIT License

Copyright (c) 2009 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma endregion
#pragma once

extern int TreeSet_COMPARE(unsigned __int32 shard, void* x, void* y);

namespace System {

	#pragma region Node Class

	template <typename T>
	class Node
	{
	public:
		__int8 isRed;
		Node* left;
		Node* right;
		T item;

		__device__ static Node* CreateNode(fallocContext* fallocCtx, T item, bool isRed /*true*/)
		{
			Node* node = falloc<Node>(fallocCtx);
			node->item = item;
			node->isRed = isRed;
			node->left = node->right = nullptr;
			return node;
		}

	};

	#pragma endregion

	template <typename T>
	class TreeSet
	{
	private:
		unsigned __int32 _shard;
		unsigned __int32 _count;
		Node<T>* _root;
		unsigned __int16 _version;
		fallocContext* _fallocCtx;

	private:
		// red calcs
		__device__ static bool IsRed(Node<T>* t) { return ((t != nullptr) && t->isRed); }
		__device__ static bool Is4Node(Node<T>* t) { return (IsRed(t->left) && IsRed(t->right)); }

		__device__ static void Split4Node(Node<T>* node)
		{
			node->isRed = true;
			node->left->isRed = false;
			node->right->isRed = false;
		}

		// rotation
		__device__ static Node<T>* RotateLeft(Node<T>* node)
		{
			Node<T>* right = node->right;
			node->right = right->left;
			right->left = node;
			return right;
		}

		__device__ static Node<T>* RotateLeftRight(Node<T>* node)
		{
			Node<T>* left = node->left;
			Node<T>* right = left->right;
			node->left = right->right;
			right->right = node;
			left->right = right->left;
			right->left = left;
			return right;
		}
 
		__device__ static Node<T>* RotateRight(Node<T>* node)
		{
			Node<T>* left = node->left;
			node->left = left->right;
			left->right = node;
			return left;
		}
 
		__device__ static Node<T>* RotateRightLeft(Node<T>* node)
		{
			Node<T>* right = node->right;
			Node<T>* left = right->left;
			node->right = left->left;
			left->left = node;
			right->left = left->right;
			left->right = right;
			return left;
		}
		
		__device__ void ReplaceChildOfNodeOrRoot(Node<T>* parent, Node<T>* child, Node<T>* newChild)
		{
			if (parent != nullptr)
			{
				if (parent->left == child)
					parent->left = newChild;
				else
					parent->right = newChild;
			}
			else
				_root = newChild;
		}

		__device__ void InsertionBalance(Node<T>* current, Node<T>* parent, Node<T>* grandParent, Node<T>* greatGrandParent)
		{
			Node<T>* node;
			bool flag = (grandParent->right == parent);
			bool flag2 = (parent->right == current);
			if (flag == flag2)
				node = (flag2 ? RotateLeft(grandParent) : RotateRight(grandParent));
			else
			{
				node = (flag2 ? RotateLeftRight(grandParent) : RotateRightLeft(grandParent));
				parent = greatGrandParent;
			}
			grandParent->isRed = true;
			node->isRed = false;
			ReplaceChildOfNodeOrRoot(greatGrandParent, grandParent, node);
		}

		//////////////////////////
		// DELETE
		typedef enum
		{
			TreeRotation_LeftRightRotation = 4,
			TreeRotation_LeftRotation = 1,
			TreeRotation_RightLeftRotation = 3,
			TreeRotation_RightRotation = 2
		} TreeRotation;

		// weird red calcs
		__device__ static bool IsBlack(Node<T>* t) { return ((t != nullptr) && !t->isRed); }
		__device__ static bool IsNullOrBlack(Node<T>* t) { return ((t != nullptr) ? !t->isRed : true); }
		__device__ static bool Is2Node(Node<T>* t) { return ((IsBlack(t) && IsNullOrBlack(t->left)) && IsNullOrBlack(t->right)); }

		__device__ static TreeRotation RotationNeeded(Node<T>* parent, Node<T>* current, Node<T>* sibling)
		{
			if (IsRed(sibling->left))
				return (parent->left == current ? TreeRotation_RightLeftRotation : TreeRotation_RightRotation);
			return (parent->left == current ? TreeRotation_LeftRotation : TreeRotation_LeftRightRotation);
		}

		__device__ static Node<T>* GetSibling(Node<T>* node, Node<T>* parent)
		{
			return (parent->left == node ? parent->right : parent->left);
		}

		// split merge
		__device__ static void Merge2Nodes(Node<T>* parent, Node<T>* child1, Node<T>* child2)
		{
			parent->isRed = false;
			child1->isRed = true;
			child2->isRed = true;
		}

	public:
		__device__ Node<T>* FindNode(T item)
		{
			int num;
			for (Node<T>* node = _root; node != nullptr; node = (num < 0 ? node->left : node->right))
			{
				num = TreeSet_COMPARE(_shard, (void*)&item, (void*)&node->item);
				if (num == 0)
					return node;
			}
			return nullptr;
		}


	public:
		//__device__ TreeSet(fallocContext* fallocCtx)
		//	: _fallocCtx(fallocCtx) { }
		__device__ void xtor(unsigned __int32 shard, fallocContext* fallocCtx)
		{
			_shard = shard;
			_fallocCtx = fallocCtx;
			_root = nullptr;
		}

		__device__ void Add(T item)
		{
			if (_root == nullptr)
			{
				_root = Node<T>::CreateNode(_fallocCtx, item, false);
				_count = 1;
			}
			else
			{
				Node<T>* root = _root;
				Node<T>* node = nullptr;
				Node<T>* grandParent = nullptr;
				Node<T>* greatGrandParent = nullptr;
				Node<T>* current;
				int num = 0;
				while (root != nullptr)
				{
					num = TreeSet_COMPARE(_shard, (void*)&item, (void*)&root->item);
					if (num == 0)
					{
						_root->isRed = false;
						thrownew(ThrowArgumentException, Argument_AddingDuplicate);
					}
					if (Is4Node(root))
					{
						Split4Node(root);
						if (IsRed(node))
							InsertionBalance(root, node, grandParent, greatGrandParent);
					}
					greatGrandParent = grandParent;
					grandParent = node;
					node = root;
					root = (num < 0 ? root->left : root->right);
				}
				current = Node<T>::CreateNode(_fallocCtx, item, true);
				if (num > 0)
					node->right = current;
				else
					node->left = current;
				if (node->isRed)
					InsertionBalance(current, node, grandParent, greatGrandParent);
				_root->isRed = false;
				_count++;
				_version++;
			}
		}

		__device__ void Clear()
		{
			_root = nullptr;
			_count = 0;
			_version++;
		}

		__device__ bool Contains(T item)
		{
			return (FindNode(item) != nullptr);
		}


	//////////////////////////
	// DELETE
	private:
		__device__ void ReplaceNode(Node<T>* match, Node<T>* parentOfMatch, Node<T>* succesor, Node<T>* parentOfSuccesor)
		{
			if (succesor == match)
				succesor = match->left;
			else
			{
				if (succesor->right != nullptr)
					succesor->right->isRed = false;
				if (parentOfSuccesor != match)
				{
					parentOfSuccesor->left = succesor->right;
					succesor->right = match->right;
				}
				succesor->left = match->left;
			}
			if (succesor != nullptr)
				succesor->isRed = match->isRed;
			ReplaceChildOfNodeOrRoot(parentOfMatch, match, succesor);
		}

	public:
		__device__ bool Remove(T item)
		{
			if (_root == nullptr)
				return false;
			Node<T>* root = _root;
			Node<T>* parent = nullptr;
			Node<T>* node3 = nullptr;
			Node<T>* match = nullptr;
			Node<T>* parentOfMatch = nullptr;
			bool flag = false;
			while (root != nullptr)
			{
				int num;
				if (Is2Node(root))
				{
					if (parent == nullptr)
						_root->isRed = true;
					else
					{
						Node<T>* sibling = GetSibling(root, parent);
						if (sibling->isRed)
						{
							if (parent->right == sibling)
								RotateLeft(parent);
							else
								RotateRight(parent);
							parent->isRed = true;
							sibling->isRed = false;
							ReplaceChildOfNodeOrRoot(node3, parent, sibling);
							node3 = sibling;
							if (parent == match)
								parentOfMatch = sibling;
							sibling = (parent->left == root ? parent->right : parent->left);
						}
						if (Is2Node(sibling))
							Merge2Nodes(parent, root, sibling);
						else
						{
							TreeRotation rotation = RotationNeeded(parent, root, sibling);
							Node<T>* newChild = nullptr;
							switch (rotation)
							{
								case TreeRotation_LeftRotation:
									sibling->right->isRed = false;
									newChild = RotateLeft(parent);
									break;
								case TreeRotation_RightRotation:
									sibling->left->isRed = false;
									newChild = RotateRight(parent);
									break;
								case TreeRotation_RightLeftRotation:
									newChild = RotateRightLeft(parent);
									break;
								case TreeRotation_LeftRightRotation:
									newChild = RotateLeftRight(parent);
									break;
							}
							newChild->isRed = parent->isRed;
							parent->isRed = false;
							root->isRed = true;
							ReplaceChildOfNodeOrRoot(node3, parent, newChild);
							if (parent == match)
								parentOfMatch = newChild;
							node3 = newChild;
						}
					}
				}
				num = (flag ? -1 : TreeSet_COMPARE(_shard, (void*)&item, (void*)&root->item));
				if (num == 0)
				{
					flag = true;
					match = root;
					parentOfMatch = parent;
				}
				node3 = parent;
				parent = root;
				root = (num < 0 ? root->left : root->right);
			}
			if (match != nullptr)
			{
				ReplaceNode(match, parentOfMatch, parent, node3);
				_count--;
			}
			if (_root != nullptr)
				_root->isRed = false;
			_version++;
			return flag;
		}

	//////////////////////////
	// ENUMERATE
	private:
		Node<T>* _current;
		T defaultT;

	public:
		T Current;

		__device__ void EnumeratorBegin(fallocContext* ctx)
		{
			_current = nullptr;
			for (Node<T>* node = _root; node != nullptr; node = node->left)
				fallocPush<Node<T>*>(ctx, node);
		}
		__device__ void EnumeratorEnd(fallocContext* ctx)
		{
			_current = nullptr;
		}

		__device__ bool EnumeratorMoveNext(fallocContext* ctx)
		{
			if (fallocAtStart(ctx))
			{
				_current = nullptr;
				Current = defaultT;
				return false;
			}
			_current = fallocPop<Node<T>*>(ctx);
			for (Node<T>* node = _current->right; node != nullptr; node = node->left)
				fallocPush<Node<T>*>(ctx, node);
			Current = _current->item;
			return true;
		}

	};

}