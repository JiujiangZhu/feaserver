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
#define COMPARE(x, y) 1 //wcscmp(x,y)

namespace System {

	#pragma region Node Class

	typedef struct _Node {
		__int8 isRed;
		void* item;
		struct _Node* left;
		struct _Node* right;
	} Node;

	__device__ static Node* CreateNode(fallocDeviceContext* deviceCtx, void* item, bool isRed /*true*/)
	{
		Node* node = (Node*)falloc(deviceCtx, sizeof(Node));
		node->item = item;
		node->isRed = isRed;
		node->left = node->right = nullptr;
		return node;
	}

	// red calcs
	__device__ static bool IsRed(Node* t) { return ((t != nullptr) && t->isRed); }
	__device__ static bool Is4Node(Node* t) { return (IsRed(t->left) && IsRed(t->right)); }

	__device__ static void Split4Node(Node* t)
	{
		t->isRed = true;
		t->left->isRed = false;
		t->right->isRed = false;
	}

	// rotation
	__device__ static Node* RotateLeft(Node* t)
	{
		Node* right = t->right;
		t->right = right->left;
		right->left = t;
		return right;
	}

	__device__ static Node* RotateLeftRight(Node* t)
	{
		Node* left = t->left;
		Node* right = left->right;
		t->left = right->right;
		right->right = t;
		left->right = right->left;
		right->left = left;
		return right;
	}
 
	__device__ static Node* RotateRight(Node* t)
	{
		Node* left = t->left;
		t->left = left->right;
		left->right = t;
		return left;
	}
 
	__device__ static Node* RotateRightLeft(Node* t)
	{
		Node* right = t->right;
		Node* left = right->left;
		t->right = left->left;
		left->left = t;
		right->left = left->right;
		left->right = right;
		return left;
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
	__device__ static bool IsBlack(Node* t) { return ((t != nullptr) && !t->isRed); }
	__device__ static bool IsNullOrBlack(Node* t) { return (t != nullptr ? !t->isRed : true); }
	__device__ static bool Is2Node(Node* t) { return ((IsBlack(t) && IsNullOrBlack(t->left)) && IsNullOrBlack(t->right)); }

	__device__ static TreeRotation RotationNeeded(Node* parent, Node* current, Node* sibling)
	{
		if (IsRed(sibling->left))
			return (parent->left == current ? TreeRotation_RightLeftRotation : TreeRotation_RightRotation);
		return (parent->left == current ? TreeRotation_LeftRotation : TreeRotation_LeftRightRotation);
	}

	__device__ static Node* GetSibling(Node* node, Node* parent)
	{
		return (parent->left == node ? parent->right : parent->left);
	}

	// split merge
	__device__ static void Merge2Nodes(Node* parent, Node* child1, Node* child2)
	{
		parent->isRed = false;
		child1->isRed = true;
		child2->isRed = true;
	}

	#pragma endregion

	template <typename T>
	class TreeSet
	{
	private:
		unsigned __int32 _count;
		Node* _root;
		unsigned __int16 _version;
		fallocDeviceContext* _deviceCtx;

	private:
		__device__ void ReplaceChildOfNodeOrRoot(Node* parent, Node* child, Node* newChild)
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

		__device__ void InsertionBalance(Node* current, Node* parent, Node* grandParent, Node* greatGrandParent)
		{
			Node* node;
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

	public:
		__device__ Node* FindNode(T item)
		{
			int num;
			for (Node* node = _root; node != nullptr; node = (num < 0 ? node->left : node->right))
			{
				num = COMPARE(item, node->Item);
				if (num == 0)
					return node;
			}
			return nullptr;
		}


	public:
		//__device__ TreeSet(fallocDeviceContext* deviceCtx)
		//	: _deviceCtx(deviceCtx) { }
		__device__ void xtor(fallocDeviceContext* deviceCtx)
		{
			_deviceCtx = deviceCtx;
		}

		__device__ void Add(T* item)
		{
			if (_root == nullptr)
			{
				_root = CreateNode(_deviceCtx, item, false);
				_count = 1;
			}
			else
			{
				Node* root = _root;
				Node* node = nullptr;
				Node* grandParent = nullptr;
				Node* greatGrandParent = nullptr;
				Node* current;
				int num = 0;
				while (root != nullptr)
				{
					num = COMPARE(item, root->Item);
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
				current = CreateNode(_deviceCtx, item, true);
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

		__device__ bool Contains(T* item)
		{
			return (FindNode(item) != nullptr);
		}


	//////////////////////////
	// DELETE
	private:
		__device__ void ReplaceNode(Node* match, Node* parentOfMatch, Node* succesor, Node* parentOfSuccesor)
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
		__device__ bool Remove(T* item)
		{
			if (_root == nullptr)
				return false;
			Node* root = _root;
			Node* parent = nullptr;
			Node* node3 = nullptr;
			Node* match = nullptr;
			Node* parentOfMatch = nullptr;
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
						Node* sibling = GetSibling(root, parent);
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
							Node* newChild = nullptr;
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
				num = (flag ? -1 : COMPARE(item, root->Item));
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

	};


	//internal bool InOrderTreeWalk(TreeWalkAction<T> action)
	//{
	//    if (this.root != nullptr)
	//    {
	//        Stack<Node<T>> stack = new Stack<Node<T>>(2 * ((int) Math.Log((double) (this.Count + 1))));
	//        Node<T> root = t->root;
	//        while (root != nullptr)
	//        {
	//            stack.Push(root);
	//            root = root.Left;
	//        }
	//        while (stack.Count != 0)
	//        {
	//            root = stack.Pop();
	//            if (!action(root))
	//                return false;
	//            for (Node<T> node2 = root.Right; node2 != nullptr; node2 = node2.Left)
	//                stack.Push(node2);
	//        }
	//    }
	//    return true;
	//}
	//

}