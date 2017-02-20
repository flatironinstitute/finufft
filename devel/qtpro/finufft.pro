QT += core network
QT -= gui

CONFIG += c++11

CONFIG -= app_bundle #Please apple, don't make a bundle

DESTDIR = bin
OBJECTS_DIR = build
MOC_DIR=build
TARGET = finufft_qtpro
TEMPLATE = app

DEFINES += NEED_EXTERN_C

HEADERS += \
    ../../src/cnufftspread.h \
    ../../src/common.h \
    ../../src/dirft.h \
    ../../src/finufft.h \
    ../../src/twopispread.h \
    ../../src/utils.h \
    ../../contrib/besseli.h \
    ../../contrib/legendre_rule_fast.h

SOURCES += \
    ../../src/cnufftspread.cpp \
    ../../src/common.cpp \
    ../../src/dirft1d.cpp \
    ../../src/dirft2d.cpp \
    ../../src/dirft3d.cpp \
    ../../src/finufft1d.cpp \
    ../../src/finufft2d.cpp \
    ../../src/finufft3d.cpp \
    ../../src/twopispread.cpp \
    ../../src/utils.cpp \
    ../../contrib/besseli.cpp \
    ../../contrib/testi0.cpp \
    ../../contrib/legendre_rule_fast.c

#FFTW
LIBS += -lfftw3 -lfftw3_threads

#OPENMP
!macx {
  QMAKE_LFLAGS += -fopenmp
  QMAKE_CXXFLAGS += -fopenmp
}
