#ifndef _SCTL_VTUDATA_
#define _SCTL_VTUDATA_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(comm.hpp)
#include SCTL_INCLUDE(vector.hpp)
#include SCTL_INCLUDE(mem_mgr.hpp)

namespace SCTL_NAMESPACE {

class Comm;
template <class ValueType> class Vector;
template <class ValueType> class Matrix;

struct VTUData {
  typedef float VTKReal;

  // Point data
  Vector<VTKReal> coord;  // always 3D
  Vector<VTKReal> value;

  // Cell data
  Vector<int32_t> connect;
  Vector<int32_t> offset;
  Vector<uint8_t> types;

  void WriteVTK(const std::string& fname, const Comm& comm) const;

  template <class ElemLst> void AddElems(const ElemLst elem_lst, Integer order, const Comm& comm = Comm::Self());
  template <class ElemLst, class ValueBasis> void AddElems(const ElemLst elem_lst, const Vector<ValueBasis>& elem_value, Integer order, const Comm& comm = Comm::Self());

  private:
    template <class CoordType, Integer ELEM_DIM> static Matrix<CoordType> VTK_Nodes(Integer order);
};

}

#include SCTL_INCLUDE(vtudata.txx)

#endif //_SCTL_VTUDATA_
