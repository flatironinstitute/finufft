#include <omp.h>
#include <cmath>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <type_traits>

#include SCTL_INCLUDE(profile.hpp)

namespace SCTL_NAMESPACE {

#ifdef SCTL_MEMDEBUG
template <class ValueType> inline ConstIterator<ValueType>::ConstIterator(const ValueType* base_, difference_type len_, bool dynamic_alloc) {
  this->base = (char*)base_;
  this->len = len_ * (Long)sizeof(ValueType);
  this->offset = 0;
  SCTL_ASSERT_MSG((uintptr_t)(this->base + this->offset) % alignof(ValueType) == 0, "invalid alignment during pointer type conversion.");
  if (dynamic_alloc) {
    MemoryManager::MemHead& mh = *&MemoryManager::GetMemHead((char*)this->base);
    MemoryManager::CheckMemHead(mh);
    alloc_ctr = mh.alloc_ctr;
    mem_head = &mh;
  } else
    mem_head = nullptr;
}

template <class ValueType> inline void ConstIterator<ValueType>::IteratorAssertChecks(Long j) const {
  //const auto& base = this->base;
  const auto& offset = this->offset + j * (Long)sizeof(ValueType);
  const auto& len = this->len;
  const auto& mem_head = this->mem_head;
  const auto& alloc_ctr = this->alloc_ctr;

  if (*this == NullIterator<ValueType>()) SCTL_WARN("dereferencing a nullptr is undefined.");
  SCTL_ASSERT_MSG(offset >= 0 && offset + (Long)sizeof(ValueType) <= len, "access to pointer [B" << (offset < 0 ? "" : "+") << offset << ",B" << (offset + (Long)sizeof(ValueType) < 0 ? "" : "+") << offset + (Long)sizeof(ValueType) << ") is outside of the range [B,B+" << len << ").");
  if (mem_head) {
    MemoryManager::MemHead& mh = *(MemoryManager::MemHead*)(mem_head);
    SCTL_ASSERT_MSG(mh.alloc_ctr == alloc_ctr, "invalid memory address or corrupted memory.");
  }
}

template <class ValueType> inline typename ConstIterator<ValueType>::reference ConstIterator<ValueType>::operator*() const {
  this->IteratorAssertChecks();
  return *(ValueType*)(base + offset);
}

template <class ValueType> inline typename ConstIterator<ValueType>::pointer ConstIterator<ValueType>::operator->() const {
  this->IteratorAssertChecks();
  return (ValueType*)(base + offset);
}

template <class ValueType> inline typename ConstIterator<ValueType>::reference ConstIterator<ValueType>::operator[](difference_type j) const {
  this->IteratorAssertChecks(j);
  return *(ValueType*)(base + offset + j * (Long)sizeof(ValueType));
}

template <class ValueType> inline typename Iterator<ValueType>::reference Iterator<ValueType>::operator*() const {
  this->IteratorAssertChecks();
  return *(ValueType*)(this->base + this->offset);
}

template <class ValueType> inline typename Iterator<ValueType>::value_type* Iterator<ValueType>::operator->() const {
  this->IteratorAssertChecks();
  return (ValueType*)(this->base + this->offset);
}

template <class ValueType> inline typename Iterator<ValueType>::reference Iterator<ValueType>::operator[](difference_type j) const {
  this->IteratorAssertChecks(j);
  return *(ValueType*)(this->base + this->offset + j * (Long)sizeof(ValueType));
}

#endif

inline MemoryManager::MemoryManager(Long N) {
  buff_size = N;
  {  // Allocate buff
    SCTL_ASSERT(SCTL_MEM_ALIGN <= 0x8000);
    Long alignment = SCTL_MEM_ALIGN - 1;
    char* base_ptr = (char*)::malloc(N + 2 + alignment);
    SCTL_ASSERT_MSG(base_ptr, "memory allocation failed.");
    buff = (char*)((uintptr_t)(base_ptr + 2 + alignment) & ~(uintptr_t)alignment);
    ((uint16_t*)buff)[-1] = (uint16_t)(buff - base_ptr);
  }
  {  // Initialize to init_mem_val
#ifdef SCTL_MEMDEBUG
#pragma omp parallel for
    for (Long i = 0; i < buff_size; i++) {
      buff[i] = init_mem_val;
    }
#endif
  }
  n_dummy_indx = new_node();
  Long n_indx = new_node();
  MemNode& n_dummy = node_buff[n_dummy_indx - 1];
  MemNode& n = node_buff[n_indx - 1];

  n_dummy.size = 0;
  n_dummy.free = false;
  n_dummy.prev = 0;
  n_dummy.next = n_indx;
  n_dummy.mem_ptr = &buff[0];
  SCTL_ASSERT(n_indx);

  n.size = N;
  n.free = true;
  n.prev = n_dummy_indx;
  n.next = 0;
  n.mem_ptr = &buff[0];
  n.it = free_map.insert(std::make_pair(N, n_indx));

  //omp_init_lock(&omp_lock);
}

inline MemoryManager::~MemoryManager() {
  Check();
  MemNode* n_dummy = &node_buff[n_dummy_indx - 1];
  MemNode* n = &node_buff[n_dummy->next - 1];
  if (!n->free || n->size != buff_size || node_stack.size() != node_buff.size() - 2 || !system_malloc.empty()) {
    SCTL_WARN("memory leak detected.");
  }
  //omp_destroy_lock(&omp_lock);

  {  // free buff
    SCTL_ASSERT(buff);
    ::free(buff - ((uint16_t*)buff)[-1]);
    buff = nullptr;
  }
}

inline MemoryManager::MemHead& MemoryManager::GetMemHead(char* I) {
  SCTL_ASSERT_MSG(I != nullptr, "nullptr exception.");
  static uintptr_t alignment = SCTL_MEM_ALIGN - 1;
  static uintptr_t header_size = (uintptr_t)(sizeof(MemHead) + alignment) & ~(uintptr_t)alignment;
  return *(MemHead*)(((char*)I) - header_size);
}

inline void MemoryManager::CheckMemHead(const MemHead& mem_head) {  // Verify header check_sum
#ifdef SCTL_MEMDEBUG
  Long check_sum = 0;
  const unsigned char* base_ = (const unsigned char*)&mem_head;
  for (Integer i = 0; i < (Integer)sizeof(MemHead); i++) {
    check_sum += base_[i];
  }
  check_sum -= mem_head.check_sum;
  check_sum = check_sum & ((1UL << (8 * sizeof(mem_head.check_sum))) - 1);
  SCTL_ASSERT_MSG(check_sum == mem_head.check_sum, "invalid memory address or corrupted memory.");
#endif
}

inline Iterator<char> MemoryManager::malloc(const Long n_elem, const Long type_size, const MemHead::TypeID type_id) const {
  if (!n_elem) return NullIterator<char>();
  static uintptr_t alignment = SCTL_MEM_ALIGN - 1;
  static uintptr_t header_size = (uintptr_t)(sizeof(MemHead) + alignment) & ~(uintptr_t)alignment;

  Long size = n_elem * type_size + header_size;
  size = (uintptr_t)(size + alignment) & ~(uintptr_t)alignment;
  char* base = nullptr;

  static Long alloc_ctr = 0;
  Long head_alloc_ctr, n_indx;
  #pragma omp critical(SCTL_MEM_MGR_CRIT)
  {
  //mutex_lock.lock();
  //omp_set_lock(&omp_lock);
  alloc_ctr++;
  head_alloc_ctr = alloc_ctr;
  std::multimap<Long, Long>::iterator it = free_map.lower_bound(size);
  n_indx = (it != free_map.end() ? it->second : 0);
  if (n_indx) {  // Allocate from buff
    Long n_free_indx = (it->first > size ? new_node() : 0);
    MemNode& n = node_buff[n_indx - 1];
    assert(n.size == it->first);
    assert(n.it == it);
    assert(n.free);

    if (n_free_indx) {  // Create a node for the remaining free part.
      MemNode& n_free = node_buff[n_free_indx - 1];
      n_free = n;
      n_free.size -= size;
      n_free.mem_ptr = (char*)n_free.mem_ptr + size;
      {  // Insert n_free to the link list
        n_free.prev = n_indx;
        if (n_free.next) {
          Long n_next_indx = n_free.next;
          MemNode& n_next = node_buff[n_next_indx - 1];
          n_next.prev = n_free_indx;
        }
        n.next = n_free_indx;
      }
      assert(n_free.free);  // Insert n_free to free map
      n_free.it = free_map.insert(std::make_pair(n_free.size, n_free_indx));
      n.size = size;  // Update n
    }

    n.free = false;
    free_map.erase(it);
    base = n.mem_ptr;
  }
  //omp_unset_lock(&omp_lock);
  //mutex_lock.unlock();
  }
  if (!base) {             // Use system malloc
    char* p = (char*)::malloc(size + 2 + alignment + end_padding);
    SCTL_ASSERT_MSG(p, "memory allocation failed.");
#ifdef SCTL_MEMDEBUG
    #pragma omp critical(SCTL_MEM_MGR_CRIT)
    {  // system_malloc.insert(p)
      //mutex_lock.lock();
      //omp_set_lock(&omp_lock);
      system_malloc.insert(p);
      //omp_unset_lock(&omp_lock);
      //mutex_lock.unlock();
    }
    {  // set p[*] to init_mem_val
#pragma omp parallel for
      for (Long i = 0; i < (Long)(size + 2 + alignment + end_padding); i++) p[i] = init_mem_val;
    }
#endif
    {  // base <-- align(p)
      base = (char*)((uintptr_t)(p + 2 + alignment) & ~(uintptr_t)alignment);
      ((uint16_t*)base)[-1] = (uint16_t)(base - p);
    }
  }

  {  // Check out-of-bounds write
#ifdef SCTL_MEMDEBUG
    if (n_indx) {
#pragma omp parallel for
      for (Long i = 0; i < size; i++) SCTL_ASSERT_MSG(base[i] == init_mem_val, "memory corruption detected.");
    }
#endif
  }

  MemHead& mem_head = *(MemHead*)base;
  {  // Set mem_head
#ifdef SCTL_MEMDEBUG
    for (Integer i = 0; i < (Integer)sizeof(MemHead); i++) base[i] = init_mem_val;
#endif
    mem_head.n_indx = n_indx;
    mem_head.n_elem = n_elem;
    mem_head.type_size = type_size;
    mem_head.alloc_ctr = head_alloc_ctr;
    mem_head.type_id = type_id;
  }
  {  // Set header check_sum
#ifdef SCTL_MEMDEBUG
    Long check_sum = 0;
    unsigned char* base_ = (unsigned char*)base;
    mem_head.check_sum = 0;
    for (Integer i = 0; i < (Integer)sizeof(MemHead); i++) check_sum += base_[i];
    check_sum = check_sum & ((1UL << (8 * sizeof(mem_head.check_sum))) - 1);
    mem_head.check_sum = check_sum;
#endif
  }
  Profile::Add_MEM(n_elem * type_size);
#ifdef SCTL_MEMDEBUG
  return Iterator<char>(base + header_size, n_elem * type_size, true);
#else
  return base + header_size;
#endif
}

inline void MemoryManager::free(Iterator<char> p) const {
  if (p == NullIterator<char>()) return;
  static uintptr_t alignment = SCTL_MEM_ALIGN - 1;
  static uintptr_t header_size = (uintptr_t)(sizeof(MemHead) + alignment) & ~(uintptr_t)alignment;
  SCTL_UNUSED(header_size);

  MemHead& mem_head = GetMemHead(&p[0]);
  Long n_indx = mem_head.n_indx;
  Long n_elem = mem_head.n_elem;
  Long type_size = mem_head.type_size;
  char* base = (char*)&mem_head;

  {  // Verify header check_sum; set array to init_mem_val
#ifdef SCTL_MEMDEBUG
    CheckMemHead(mem_head);
    Long size = mem_head.n_elem * mem_head.type_size;
#pragma omp parallel for
    for (Long i = 0; i < size; i++) p[i] = init_mem_val;
    for (Integer i = 0; i < (Integer)sizeof(MemHead); i++) base[i] = init_mem_val;
#endif
  }

  if (n_indx == 0) {  // Use system free
    assert(base < &buff[0] || base >= &buff[buff_size]);
    char* p_;
    {  // p_ <-- unalign(base)
      p_ = (char*)((uintptr_t)base - ((uint16_t*)base)[-1]);
    }
#ifdef SCTL_MEMDEBUG
    {  // Check out-of-bounds write
      base[-1] = init_mem_val;
      base[-2] = init_mem_val;

      Long size = n_elem * type_size + header_size;
      size = (uintptr_t)(size + alignment) & ~(uintptr_t)alignment;
#pragma omp parallel for
      for (Long i = 0; i < (Long)(size + 2 + alignment + end_padding); i++) {
        SCTL_ASSERT_MSG(p_[i] == init_mem_val, "memory corruption detected.");
      }
    }
    #pragma omp critical(SCTL_MEM_MGR_CRIT)
    if (buff != nullptr) {  // system_malloc.erase(p_)
      //mutex_lock.lock();
      //omp_set_lock(&omp_lock);
      SCTL_ASSERT_MSG(system_malloc.erase(p_) == 1, "double free or corruption.");
      //omp_unset_lock(&omp_lock);
      //mutex_lock.unlock();
    }
#endif
    ::free(p_);
  } else {
#ifdef SCTL_MEMDEBUG
    {  // Check out-of-bounds write
      MemNode& n = node_buff[n_indx - 1];
      char* base = n.mem_ptr;
#pragma omp parallel for
      for (Long i = 0; i < n.size; i++) {
        SCTL_ASSERT_MSG(base[i] == init_mem_val, "memory corruption detected.");
      }
    }
#endif
    assert(n_indx <= (Long)node_buff.size());
    #pragma omp critical(SCTL_MEM_MGR_CRIT)
    {
    //mutex_lock.lock();
    //omp_set_lock(&omp_lock);
    MemNode& n = node_buff[n_indx - 1];
    assert(!n.free && n.size > 0 && n.mem_ptr == base);
    if (n.prev != 0 && node_buff[n.prev - 1].free) {
      Long n_prev_indx = n.prev;
      MemNode& n_prev = node_buff[n_prev_indx - 1];
      n.size += n_prev.size;
      n.mem_ptr = n_prev.mem_ptr;
      n.prev = n_prev.prev;
      free_map.erase(n_prev.it);
      delete_node(n_prev_indx);

      if (n.prev) {
        node_buff[n.prev - 1].next = n_indx;
      }
    }
    if (n.next != 0 && node_buff[n.next - 1].free) {
      Long n_next_indx = n.next;
      MemNode& n_next = node_buff[n_next_indx - 1];
      n.size += n_next.size;
      n.next = n_next.next;
      free_map.erase(n_next.it);
      delete_node(n_next_indx);

      if (n.next) {
        node_buff[n.next - 1].prev = n_indx;
      }
    }
    n.free = true;  // Insert n to free_map
    n.it = free_map.insert(std::make_pair(n.size, n_indx));
    //omp_unset_lock(&omp_lock);
    //mutex_lock.unlock();
    }
  }
  Profile::Add_MEM(-n_elem * type_size);
}

inline void MemoryManager::print() const {
  if (!buff_size) return;
  #pragma omp critical(SCTL_MEM_MGR_CRIT)
  {
  //mutex_lock.lock();
  //omp_set_lock(&omp_lock);

  Long size = 0;
  Long largest_size = 0;
  MemNode* n = &node_buff[n_dummy_indx - 1];
  std::cout << "\n|";
  while (n->next) {
    n = &node_buff[n->next - 1];
    if (n->free) {
      std::cout << ' ';
      largest_size = std::max(largest_size, n->size);
    } else {
      std::cout << '#';
      size += n->size;
    }
  }
  std::cout << "|  allocated=" << round(size * 1000.0 / buff_size) / 10 << "%";
  std::cout << "  largest_free=" << round(largest_size * 1000.0 / buff_size) / 10 << "%\n";

  //omp_unset_lock(&omp_lock);
  //mutex_lock.unlock();
  }
}

inline void MemoryManager::test() {
  Long M = 2000000000;
  {  // With memory manager
    Long N = (Long)(M * sizeof(double) * 1.1);
    double tt;
    Iterator<double> tmp;

    std::cout << "With memory manager: ";
    MemoryManager memgr(N);

    for (Integer j = 0; j < 3; j++) {
      tmp = (Iterator<double>)memgr.malloc(M * sizeof(double));
      SCTL_ASSERT(tmp != NullIterator<double>());
      tt = omp_get_wtime();
#pragma omp parallel for
      for (Long i = 0; i < M; i += 64) tmp[i] = (double)i;
      tt = omp_get_wtime() - tt;
      std::cout << tt << ' ';
      memgr.free((Iterator<char>)tmp);
    }
    std::cout << '\n';
  }
  {  // Without memory manager
    double tt;
    double* tmp;

    std::cout << "Without memory manager: ";
    for (Integer j = 0; j < 3; j++) {
      tmp = (double*)::malloc(M * sizeof(double));
      SCTL_ASSERT(tmp != nullptr);
      tt = omp_get_wtime();
#pragma omp parallel for
      for (Long i = 0; i < M; i += 64) tmp[i] = (double)i;
      tt = omp_get_wtime() - tt;
      std::cout << tt << ' ';
      ::free(tmp);
    }
    std::cout << '\n';
  }
}

inline void MemoryManager::Check() const {
#ifdef SCTL_MEMDEBUG
  // print();
  #pragma omp critical(SCTL_MEM_MGR_CRIT)
  {
  //mutex_lock.lock();
  //omp_set_lock(&omp_lock);
  MemNode* curr_node = &node_buff[n_dummy_indx - 1];
  while (curr_node->next) {
    curr_node = &node_buff[curr_node->next - 1];
    if (curr_node->free) {
      char* base = curr_node->mem_ptr;
#pragma omp parallel for
      for (Long i = 0; i < curr_node->size; i++) {
        SCTL_ASSERT_MSG(base[i] == init_mem_val, "memory corruption detected.");
      }
    }
  }
  //omp_unset_lock(&omp_lock);
  //mutex_lock.unlock();
  }
#endif
}

inline Long MemoryManager::new_node() const {
  if (node_stack.empty()) {
    node_buff.resize(node_buff.size() + 1);
    node_stack.push(node_buff.size());
  }

  Long indx = node_stack.top();
  node_stack.pop();
  assert(indx);
  return indx;
}

inline void MemoryManager::delete_node(Long indx) const {
  assert(indx);
  assert(indx <= (Long)node_buff.size());
  MemNode& n = node_buff[indx - 1];
  n.free = false;
  n.size = 0;
  n.prev = 0;
  n.next = 0;
  n.mem_ptr = nullptr;
  node_stack.push(indx);
}

template <class ValueType> inline Iterator<ValueType> aligned_new(Long n_elem, const MemoryManager* mem_mgr) {
  if (!n_elem) return NullIterator<ValueType>();

  static MemoryManager def_mem_mgr(0);
  if (!mem_mgr) mem_mgr = &def_mem_mgr;
  Iterator<ValueType> A = (Iterator<ValueType>)mem_mgr->malloc(n_elem, sizeof(ValueType), typeid(ValueType).hash_code());
  SCTL_ASSERT_MSG(A != NullIterator<ValueType>(), "memory allocation failed.");

  if (!std::is_trivial<ValueType>::value) {  // Call constructors
                                          // printf("%s\n", __PRETTY_FUNCTION__);
#pragma omp parallel for schedule(static)
    for (Long i = 0; i < n_elem; i++) {
      ValueType* Ai = new (&A[i]) ValueType();
      assert(Ai == (&A[i]));
      SCTL_UNUSED(Ai);
    }
  } else {
#ifdef SCTL_MEMDEBUG
    static Long random_init_val = 1;
    Iterator<char> A_ = (Iterator<char>)A;
#pragma omp parallel for schedule(static)
    for (Long i = 0; i < n_elem * (Long)sizeof(ValueType); i++) {
      A_[i] = random_init_val + i;
    }
    random_init_val += n_elem * sizeof(ValueType);
#endif
  }

  return A;
}

template <class ValueType> inline void aligned_delete(Iterator<ValueType> A, const MemoryManager* mem_mgr) {
  if (A == NullIterator<ValueType>()) return;

  if (!std::is_trivial<ValueType>::value) {  // Call destructors
    // printf("%s\n", __PRETTY_FUNCTION__);
    MemoryManager::MemHead& mem_head = MemoryManager::GetMemHead((char*)&A[0]);
#ifdef SCTL_MEMDEBUG
    MemoryManager::CheckMemHead(mem_head);
    SCTL_ASSERT_MSG(mem_head.type_id==typeid(ValueType).hash_code(), "pointer to aligned_delete has different type than what was used in aligned_new.");
#endif
    Long n_elem = mem_head.n_elem;
    for (Long i = 0; i < n_elem; i++) {
      A[i].~ValueType();
    }
  } else {
#ifdef SCTL_MEMDEBUG
    MemoryManager::MemHead& mem_head = MemoryManager::GetMemHead((char*)&A[0]);
    MemoryManager::CheckMemHead(mem_head);
    SCTL_ASSERT_MSG(mem_head.type_id==typeid(ValueType).hash_code(), "pointer to aligned_delete has different type than what was used in aligned_new.");
    Long size = mem_head.n_elem * mem_head.type_size;
    Iterator<char> A_ = (Iterator<char>)A;
#pragma omp parallel for
    for (Long i = 0; i < size; i++) {
      A_[i] = 0;
    }
#endif
  }

  static MemoryManager def_mem_mgr(0);
  if (!mem_mgr) mem_mgr = &def_mem_mgr;
  mem_mgr->free((Iterator<char>)A);
}

template <class ValueType> inline Iterator<ValueType> memcopy(Iterator<ValueType> destination, ConstIterator<ValueType> source, Long num) {
  if (destination != source && num) {
#ifdef SCTL_MEMDEBUG
    SCTL_UNUSED(destination[num - 1]);
    SCTL_UNUSED(source[num - 1]     );
#endif
    if (std::is_trivially_copyable<ValueType>::value) {
      memcpy((void*)&destination[0], (const void*)&source[0], num * sizeof(ValueType));
    } else {
      for (Long i = 0; i < num; i++) destination[i] = source[i];
    }
  }
  return destination;
}

template <class ValueType> inline Iterator<ValueType> memset(Iterator<ValueType> ptr, int value, Long num) {
  if (num) {
#ifdef SCTL_MEMDEBUG
    SCTL_UNUSED(ptr[0]      );
    SCTL_UNUSED(ptr[num - 1]);
#endif
    SCTL_ASSERT(std::is_trivially_copyable<ValueType>::value);
    ::memset((void*)&ptr[0], value, num * sizeof(ValueType));
  }
  return ptr;
}

}  // end namespace
