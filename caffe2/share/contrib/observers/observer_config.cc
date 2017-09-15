#include "caffe2/share/contrib/observers/observer_config.h"

namespace caffe2 {

int QPLConfig::netSampleRate_ = 1;
int QPLConfig::operatorNetSampleRatio_ = 0;
int QPLConfig::skipIters_ = 10;

unique_ptr<ObserverReporter> QPLConfig::reporter_ = nullptr;
}
