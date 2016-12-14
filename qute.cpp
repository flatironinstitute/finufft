#include "qute.h"
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

using namespace std;

void QTime::start()
{
    gettimeofday(&initial, 0);
}

int QTime::restart()
{
    int delta = this->elapsed();
    this->start();
    return delta;
}

int QTime::elapsed()
{
    struct timeval now;
    gettimeofday(&now, 0);
    int delta = 1000 * (now.tv_sec - (initial.tv_sec + 1));
    delta += (now.tv_usec + (1000000 - initial.tv_usec)) / 1000;
    return delta;
}

int qrand()
{
    return rand();
}

class QStringPrivate {
public:
    QString *q;
};

QString::QString() {
    d=new QStringPrivate;
    d->q=this;
}

QString::~QString()
{
    delete d;
}


