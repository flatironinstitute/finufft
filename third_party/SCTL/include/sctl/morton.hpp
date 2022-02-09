#ifndef _SCTL_MORTON_
#define _SCTL_MORTON_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(math_utils.hpp)
#include <cstdint>

#ifndef SCTL_MAX_DEPTH
#define SCTL_MAX_DEPTH 15
#endif

namespace SCTL_NAMESPACE {

template <class ValueType> class Vector;

template <Integer DIM = 3> class Morton {

 public:
  #if SCTL_MAX_DEPTH < 7
  typedef uint8_t UINT_T;
  #elif SCTL_MAX_DEPTH < 15
  typedef uint16_t UINT_T;
  #elif SCTL_MAX_DEPTH < 31
  typedef uint32_t UINT_T;
  #elif SCTL_MAX_DEPTH < 63
  typedef uint64_t UINT_T;
  #endif

  static constexpr Integer MAX_DEPTH = SCTL_MAX_DEPTH;

  static constexpr Integer MaxDepth() {
    return MAX_DEPTH;
  }

  Morton() {
    depth = 0;
    for (Integer i = 0; i < DIM; i++) x[i] = 0;
  }

  template <class T> explicit Morton(ConstIterator<T> coord, uint8_t depth_ = MAX_DEPTH) {
    depth = depth_;
    SCTL_ASSERT(depth <= MAX_DEPTH);
    UINT_T mask = ~((((UINT_T)1) << (MAX_DEPTH - depth)) - 1);
    for (Integer i = 0; i < DIM; i++) x[i] = mask & (UINT_T)floor((double)coord[i] * maxCoord);
  }

  uint8_t Depth() const { return depth; }

  template <class T> void Coord(Iterator<T> coord) const {
    static const T s = 1.0 / ((T)maxCoord);
    for (Integer i = 0; i < DIM; i++) coord[i] = (T)x[i] * s;
  }

  Morton Next() const {
    UINT_T mask = ((UINT_T)1) << (MAX_DEPTH - depth);
    Integer d, i;

    Morton m = *this;
    for (d = depth; d >= 0; d--) {
      for (i = 0; i < DIM; i++) {
        m.x[i] = (m.x[i] ^ mask);
        if ((m.x[i] & mask)) break;
      }
      if (i < DIM) break;
      mask = (mask << 1);
    }

    if (d < 0) d = 0;
    m.depth = (uint8_t)d;

    return m;
  }

  Morton Ancestor(uint8_t ancestor_level) const {
    UINT_T mask = ~((((UINT_T)1) << (MAX_DEPTH - ancestor_level)) - 1);

    Morton m;
    for (Integer i = 0; i < DIM; i++) m.x[i] = x[i] & mask;
    m.depth = ancestor_level;
    return m;
  }

  /**
   * \brief Returns the deepest first descendant.
   */
  Morton DFD(uint8_t level = MAX_DEPTH) const {
    Morton m = *this;
    m.depth = level;
    return m;
  }

  void NbrList(Vector<Morton> &nlst, uint8_t level, bool periodic) const {
    static constexpr Integer MAX_NBRS = sctl::pow<DIM,Integer>(3);
    StaticArray<Morton<DIM>,MAX_NBRS> nbrs;
    Integer Nnbrs = 0;

    UINT_T mask = ~((((UINT_T)1) << (MAX_DEPTH - level)) - 1);
    for (Integer i = 0; i < DIM; i++) nbrs[0].x[i] = x[i] & mask;
    nbrs[0].depth = level;
    Nnbrs++;

    Morton m;
    Integer k = 1;
    mask = (((UINT_T)1) << (MAX_DEPTH - level));
    if (periodic) {
      for (Integer i = 0; i < DIM; i++) {
        for (Integer j = 0; j < k; j++) {
          m = nbrs[j];
          m.x[i] = (m.x[i] + mask) & (maxCoord - 1);
          nbrs[Nnbrs] = m;
          Nnbrs++;
        }
        for (Integer j = 0; j < k; j++) {
          m = nbrs[j];
          m.x[i] = (m.x[i] - mask) & (maxCoord - 1);
          nbrs[Nnbrs] = m;
          Nnbrs++;
        }
        k = Nnbrs;
      }
    } else {
      for (Integer i = 0; i < DIM; i++) {
        for (Integer j = 0; j < k; j++) {
          m = nbrs[j];
          if (m.x[i] + mask < maxCoord) {
            m.x[i] += mask;
            nbrs[Nnbrs] = m;
            Nnbrs++;
          }
        }
        for (Integer j = 0; j < k; j++) {
          m = nbrs[j];
          if (m.x[i] >= mask) {
            m.x[i] -= mask;
            nbrs[Nnbrs] = m;
            Nnbrs++;
          }
        }
        k = Nnbrs;
      }
    }
    if (nlst.Dim() != Nnbrs) {
      nlst.ReInit(Nnbrs, nbrs);
    } else {
      for (Integer i = 0; i < Nnbrs; i++) {
        nlst[i] = nbrs[i];
      }
    }
  }

  void Children(Vector<Morton> &nlst) const {
    static const Integer cnt = (1UL << DIM);
    if (nlst.Dim() != cnt) nlst.ReInit(cnt);

    for (Integer i = 0; i < DIM; i++) nlst[0].x[i] = x[i];
    nlst[0].depth = (uint8_t)(depth + 1);

    Integer k = 1;
    UINT_T mask = (((UINT_T)1) << (MAX_DEPTH - (depth + 1)));
    for (Integer i = 0; i < DIM; i++) {
      for (Integer j = 0; j < k; j++) {
        nlst[j + k] = nlst[j];
        nlst[j + k].x[i] += mask;
      }
      k = (k << 1);
    }
  }

  bool operator<(const Morton &m) const {
    UINT_T diff = 0;
    for (Integer i = 0; i < DIM; i++) diff = diff | (x[i] ^ m.x[i]);
    if (!diff) return depth < m.depth;

    UINT_T mask = 1;
    for (Integer i = 4 * sizeof(UINT_T); i > 0; i = (i >> 1)) {
      UINT_T mask_ = (mask << i);
      if (mask_ <= diff) mask = mask_;
    }

    for (Integer i = DIM - 1; i >= 0; i--) {
      if (mask & (x[i] ^ m.x[i])) return x[i] < m.x[i];
    }
    return false; // TODO: check
  }

  bool operator>(const Morton &m) const { return m < (*this); }

  bool operator!=(const Morton &m) const {
    for (Integer i = 0; i < DIM; i++)
      if (x[i] != m.x[i]) return true;
    return (depth != m.depth);
  }

  bool operator==(const Morton &m) const { return !(*this != m); }

  bool operator<=(const Morton &m) const { return !(*this > m); }

  bool operator>=(const Morton &m) const { return !(*this < m); }

  bool isAncestor(Morton const &descendant) const { return descendant.depth > depth && descendant.Ancestor(depth) == *this; }

  Long operator-(const Morton<DIM> &I) const {
    // Intersecting -1
    // Touching 0

    Long offset0 = 1 << (MAX_DEPTH - depth - 1);
    Long offset1 = 1 << (MAX_DEPTH - I.depth - 1);

    Long diff = 0;
    for (Integer i = 0; i < DIM; i++) {
      diff = std::max<Long>(diff, abs(((Long)x[i] + offset0) - ((Long)I.x[i] + offset1)));
    }
    if (diff < offset0 + offset1) return -1;
    Integer max_depth = std::max(depth, I.depth);
    diff = (diff - offset0 - offset1) >> (MAX_DEPTH - max_depth);
    return diff;
  }

  friend std::ostream &operator<<(std::ostream &out, const Morton &mid) {
    double a = 0;
    double s = 1u << DIM;
    for (Integer j = MAX_DEPTH; j >= 0; j--) {
      for (Integer i = DIM - 1; i >= 0; i--) {
        s = s * 0.5;
        if (mid.x[i] & (((UINT_T)1) << j)) a += s;
      }
    }
    out << "(";
    for (Integer i = 0; i < DIM; i++) {
      out << mid.x[i] * 1.0 / maxCoord << ",";
    }
    out << (int)mid.depth << "," << a << ")";
    return out;
  }

 private:
  static constexpr UINT_T maxCoord = ((UINT_T)1) << (MAX_DEPTH);

  // StaticArray<UINT_T,DIM> x;
  UINT_T x[DIM];
  uint8_t depth;
};
}

#endif  //_SCTL_MORTON_
