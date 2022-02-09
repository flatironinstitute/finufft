// TODO: Implement fast stack allocation.
#ifndef _SCTL_MEM_MGR_HPP_
#define _SCTL_MEM_MGR_HPP_

//#include <mutex>

#include <sctl/common.hpp>

#include <omp.h>
#include <typeinfo>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <vector>
#include <stack>
#include <map>
#include <set>

namespace SCTL_NAMESPACE {

class MemoryManager;

#ifdef SCTL_MEMDEBUG

template <class ValueType> class ConstIterator {

  template <typename T> friend class ConstIterator;

  template <typename T> friend class Iterator;

  void IteratorAssertChecks(Long j = 0) const;

 public:
  typedef Long difference_type;
  typedef ValueType value_type;
  typedef const ValueType* pointer;
  typedef const ValueType& reference;
  typedef std::random_access_iterator_tag iterator_category;

 protected:
  char* base;
  difference_type len, offset;
  Long alloc_ctr;
  void* mem_head;
  static const Long ValueSize = sizeof(ValueType);

 public:
  ConstIterator() : base(nullptr), len(0), offset(0), alloc_ctr(0), mem_head(nullptr) {}

  // template <size_t LENGTH> ConstIterator(ValueType (&base_)[LENGTH]) {  // DEPRECATED (because mem_head cannot be stored)
  //   SCTL_ASSERT(false);
  // }

  ConstIterator(pointer base_, difference_type len_, bool dynamic_alloc = false);

  template <class AnotherType> explicit ConstIterator(const ConstIterator<AnotherType>& I) {
    this->base = I.base;
    this->len = I.len;
    this->offset = I.offset;
    SCTL_ASSERT_MSG((uintptr_t)(this->base + this->offset) % alignof(ValueType) == 0, "invalid alignment during pointer type conversion.");
    this->alloc_ctr = I.alloc_ctr;
    this->mem_head = I.mem_head;
  }

  // value_type* like operators
  reference operator*() const;

  pointer operator->() const;

  reference operator[](difference_type off) const;

  // Increment / Decrement
  ConstIterator& operator++() {
    offset += (Long)sizeof(ValueType);
    return *this;
  }

  ConstIterator operator++(int) {
    ConstIterator<ValueType> tmp(*this);
    ++*this;
    return tmp;
  }

  ConstIterator& operator--() {
    offset -= (Long)sizeof(ValueType);
    return *this;
  }

  ConstIterator operator--(int) {
    ConstIterator<ValueType> tmp(*this);
    --*this;
    return tmp;
  }

  // Arithmetic
  ConstIterator& operator+=(difference_type i) {
    offset += i * (Long)sizeof(ValueType);
    return *this;
  }

  ConstIterator operator+(difference_type i) const {
    ConstIterator<ValueType> tmp(*this);
    tmp.offset += i * (Long)sizeof(ValueType);
    return tmp;
  }

  friend ConstIterator operator+(difference_type i, const ConstIterator& right) { return (right + i); }

  ConstIterator& operator-=(difference_type i) {
    offset -= i * (Long)sizeof(ValueType);
    return *this;
  }

  ConstIterator operator-(difference_type i) const {
    ConstIterator<ValueType> tmp(*this);
    tmp.offset -= i * (Long)sizeof(ValueType);
    return tmp;
  }

  difference_type operator-(const ConstIterator& I) const {
    // if (base != I.base) SCTL_WARN("comparing two unrelated memory addresses.");
    Long diff = ((pointer)(base + offset)) - ((pointer)(I.base + I.offset));
    SCTL_ASSERT_MSG(I.base + I.offset + diff * (Long)sizeof(ValueType) == base + offset, "invalid memory address alignment.");
    return diff;
  }

  // Comparison operators
  bool operator==(const ConstIterator& I) const { return (base + offset == I.base + I.offset); }

  bool operator!=(const ConstIterator& I) const { return !(*this == I); }

  bool operator<(const ConstIterator& I) const {
    // if (base != I.base) SCTL_WARN("comparing two unrelated memory addresses.");
    return (base + offset) < (I.base + I.offset);
  }

  bool operator<=(const ConstIterator& I) const {
    // if (base != I.base) SCTL_WARN("comparing two unrelated memory addresses.");
    return (base + offset) <= (I.base + I.offset);
  }

  bool operator>(const ConstIterator& I) const {
    // if (base != I.base) SCTL_WARN("comparing two unrelated memory addresses.");
    return (base + offset) > (I.base + I.offset);
  }

  bool operator>=(const ConstIterator& I) const {
    // if (base != I.base) SCTL_WARN("comparing two unrelated memory addresses.");
    return (base + offset) >= (I.base + I.offset);
  }

  friend std::ostream& operator<<(std::ostream& out, const ConstIterator& I) {
    out << "(" << (long long)I.base << "+" << I.offset << ":" << I.len << ")";
    return out;
  }
};

template <class ValueType> class Iterator : public ConstIterator<ValueType> {

 public:
  typedef Long difference_type;
  typedef ValueType value_type;
  typedef ValueType* pointer;
  typedef ValueType& reference;
  typedef std::random_access_iterator_tag iterator_category;

 public:
  Iterator() : ConstIterator<ValueType>() {}

  //template <size_t LENGTH> Iterator(ValueType (&base_)[LENGTH]) : ConstIterator<ValueType>(base_) {}

  Iterator(pointer base_, difference_type len_, bool dynamic_alloc = false) : ConstIterator<ValueType>(base_, len_, dynamic_alloc) {}

  template <class AnotherType> explicit Iterator(const ConstIterator<AnotherType>& I) : ConstIterator<ValueType>(I) {}

  // value_type* like operators
  reference operator*() const;

  value_type* operator->() const;

  reference operator[](difference_type off) const;

  // Increment / Decrement
  Iterator& operator++() {
    this->offset += (Long)sizeof(ValueType);
    return *this;
  }

  Iterator operator++(int) {
    Iterator<ValueType> tmp(*this);
    ++*this;
    return tmp;
  }

  Iterator& operator--() {
    this->offset -= (Long)sizeof(ValueType);
    return *this;
  }

  Iterator operator--(int) {
    Iterator<ValueType> tmp(*this);
    --*this;
    return tmp;
  }

  // Arithmetic
  Iterator& operator+=(difference_type i) {
    this->offset += i * (Long)sizeof(ValueType);
    return *this;
  }

  Iterator operator+(difference_type i) const {
    Iterator<ValueType> tmp(*this);
    tmp.offset += i * (Long)sizeof(ValueType);
    return tmp;
  }

  friend Iterator operator+(difference_type i, const Iterator& right) { return (right + i); }

  Iterator& operator-=(difference_type i) {
    this->offset -= i * (Long)sizeof(ValueType);
    return *this;
  }

  Iterator operator-(difference_type i) const {
    Iterator<ValueType> tmp(*this);
    tmp.offset -= i * (Long)sizeof(ValueType);
    return tmp;
  }

  difference_type operator-(const ConstIterator<ValueType>& I) const { return static_cast<const ConstIterator<ValueType>&>(*this) - I; }
};

template <class ValueType, Long DIM> class StaticArray {
  typedef Long difference_type;

 public:
  StaticArray() = default;

  StaticArray(const StaticArray&) = default;

  StaticArray& operator=(const StaticArray&) = default;

  explicit StaticArray(std::initializer_list<ValueType> arr_) : StaticArray() {
    // static_assert(arr_.size() <= DIM, "too many initializer values"); // allowed in C++14
    SCTL_ASSERT_MSG(arr_.size() <= DIM, "too many initializer values");
    for (Long i = 0; i < (Long)arr_.size(); i++) (*this)[i] = arr_.begin()[i];
  }

  ~StaticArray() = default;

  // value_type* like operators
  const ValueType& operator*() const { return (*this)[0]; }

  ValueType& operator*() { return (*this)[0]; }

  const ValueType* operator->() const { return (ConstIterator<ValueType>)*this; }

  ValueType* operator->() { return (Iterator<ValueType>)*this; }

  const ValueType& operator[](difference_type off) const { return ((ConstIterator<ValueType>)*this)[off]; }

  ValueType& operator[](difference_type off) { return ((Iterator<ValueType>)*this)[off]; }

  operator ConstIterator<ValueType>() const { return ConstIterator<ValueType>(arr_, DIM); }

  operator Iterator<ValueType>() { return Iterator<ValueType>(arr_, DIM); }

  // Arithmetic
  ConstIterator<ValueType> operator+(difference_type i) const { return (ConstIterator<ValueType>)*this + i; }

  Iterator<ValueType> operator+(difference_type i) { return (Iterator<ValueType>)*this + i; }

  friend ConstIterator<ValueType> operator+(difference_type i, const StaticArray& right) { return i + (ConstIterator<ValueType>)right; }

  friend Iterator<ValueType> operator+(difference_type i, StaticArray& right) { return i + (Iterator<ValueType>)right; }

  ConstIterator<ValueType> operator-(difference_type i) const { return (ConstIterator<ValueType>)*this - i; }

  Iterator<ValueType> operator-(difference_type i) { return (Iterator<ValueType>)*this - i; }

  difference_type operator-(const ConstIterator<ValueType>& I) const { return (ConstIterator<ValueType>)*this - (ConstIterator<ValueType>)I; }

  // Comparison operators
  bool operator==(const ConstIterator<ValueType>& I) const { return (ConstIterator<ValueType>)*this == I; }

  bool operator!=(const ConstIterator<ValueType>& I) const { return (ConstIterator<ValueType>)*this != I; }

  bool operator< (const ConstIterator<ValueType>& I) const { return (ConstIterator<ValueType>)*this <  I; }

  bool operator<=(const ConstIterator<ValueType>& I) const { return (ConstIterator<ValueType>)*this <= I; }

  bool operator> (const ConstIterator<ValueType>& I) const { return (ConstIterator<ValueType>)*this >  I; }

  bool operator>=(const ConstIterator<ValueType>& I) const { return (ConstIterator<ValueType>)*this >= I; }

 private:

  ValueType arr_[DIM];
};

template <class ValueType> Iterator<ValueType> Ptr2Itr(void* ptr, Long len) { return Iterator<ValueType>((ValueType*)ptr, len); }
template <class ValueType> ConstIterator<ValueType> Ptr2ConstItr(const void* ptr, Long len) { return ConstIterator<ValueType>((ValueType*)ptr, len); }

#else

// StaticArray - forward declared in common.hpp

template <class ValueType> Iterator<ValueType> Ptr2Itr(void* ptr, Long len) { return (Iterator<ValueType>) ptr; }
template <class ValueType> ConstIterator<ValueType> Ptr2ConstItr(const void* ptr, Long len) { return (ConstIterator<ValueType>) ptr; }

#endif

template <class ValueType> Iterator<ValueType> NullIterator() { return Ptr2Itr<ValueType>(nullptr, 0); }

/**
 * \brief MemoryManager class declaration.
 */
class MemoryManager {

 public:
  static constexpr char init_mem_val = 42;
  static constexpr Long end_padding = 64;

  /**
   * \brief Header data for each memory block.
   */
  struct MemHead {
    typedef decltype(typeid(char).hash_code()) TypeID;
    Long n_indx;
    Long n_elem;
    Long type_size;
    Long alloc_ctr;
    TypeID type_id;
    #ifdef SCTL_MEMDEBUG
    unsigned char check_sum;
    #endif
  };

  /**
   * \brief Constructor for MemoryManager.
   */
  explicit MemoryManager(Long N);

  /**
   * \brief Constructor for MemoryManager.
   */
  ~MemoryManager();

  static MemHead& GetMemHead(char* p);

  static void CheckMemHead(const MemHead& p);

  Iterator<char> malloc(const Long n_elem, const Long type_size = sizeof(char), const MemHead::TypeID type_id = typeid(char).hash_code()) const;

  void free(Iterator<char> p) const;

  void print() const;

  static void test();

  // Check all free memory equals init_mem_val
  void Check() const;

  // A global MemoryManager object. This is the default for aligned_new and aligned_free
  static MemoryManager& glbMemMgr() {
    static MemoryManager m(SCTL_GLOBAL_MEM_BUFF * 1024LL * 1024LL);
    return m;
  }

 private:
  // Private constructor
  MemoryManager();

  // Private copy constructor
  MemoryManager(const MemoryManager& m);

  /**
   * \brief Node structure for a doubly linked list, representing free and
   * occupied memory blocks. Blocks are split, merged or state is changed
   * between free and occupied in O(1) time given the pointer to the MemNode.
   */
  struct MemNode {
    bool free;
    Long size;
    char* mem_ptr;
    Long prev, next;
    std::multimap<Long, Long>::iterator it;
  };

  /**
   * \brief Return index of one of the available MemNodes from node_stack or
   * create new MemNode by resizing node_buff.
   */
  Long new_node() const;

  /**
   * \brief Add node index for now available MemNode to node_stack.
   */
  void delete_node(Long indx) const;

  char* buff;         // pointer to memory buffer.
  Long buff_size;     // total buffer size in bytes.
  Long n_dummy_indx;  // index of first (dummy) MemNode in link list.

  mutable std::vector<MemNode> node_buff;      // storage for MemNode objects, this can only grow.
  mutable std::stack<Long> node_stack;         // stack of available free MemNodes from node_buff.
  mutable std::multimap<Long, Long> free_map;  // pair (MemNode.size, MemNode_id) for all free MemNodes.
  //mutable omp_lock_t omp_lock;                 // openmp lock to prevent concurrent changes.
  //mutable std::mutex mutex_lock;
  mutable std::set<void*> system_malloc;       // track pointers allocated using system malloc.
};

inline uintptr_t align_ptr(uintptr_t ptr) {
  const uintptr_t ALIGN_MINUS_ONE = SCTL_MEM_ALIGN - 1;
  const uintptr_t NOT_ALIGN_MINUS_ONE = ~ALIGN_MINUS_ONE;
  return ((ptr + ALIGN_MINUS_ONE) & NOT_ALIGN_MINUS_ONE);
}

/**
 * \brief Aligned allocation as an alternative to new. Uses placement new to
 * construct objects.
 */
template <class ValueType> Iterator<ValueType> aligned_new(Long n_elem = 1, const MemoryManager* mem_mgr = &MemoryManager::glbMemMgr());

/**
 * \brief Aligned de-allocation as an alternative to delete. Calls the object
 * destructor.
 */
template <class ValueType> void aligned_delete(Iterator<ValueType> A, const MemoryManager* mem_mgr = &MemoryManager::glbMemMgr());

/**
 * \brief Wrapper to memcpy. Also checks if source and destination pointers are
 * the same.
 */
template <class ValueType> Iterator<ValueType> memcopy(Iterator<ValueType> destination, ConstIterator<ValueType> source, Long num);

template <class ValueType> Iterator<ValueType> memset(Iterator<ValueType> ptr, int value, Long num);

}  // end namespace SCTL_NAMESPACE

#include SCTL_INCLUDE(mem_mgr.txx)

#endif  //_SCTL_MEM_MGR_HPP_
