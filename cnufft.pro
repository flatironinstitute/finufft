QT -= core
QT -= gui
CONFIG -= app_bundle #Please apple, don't make a bundle
CONFIG += c++11 console

DESTDIR = ./bin
OBJECTS_DIR = ./build
MOC_DIR = ./build
TARGET = cnufft
TEMPLATE = app

SOURCES += cnufftmain.cpp \
    cnufftspread.cpp besseli.cpp qute.cpp

HEADERS += \
    cnufftspread.h besseli.h qute.h

HEADERS += cnufftspread_c.h
SOURCES += cnufftspread_c.cpp

