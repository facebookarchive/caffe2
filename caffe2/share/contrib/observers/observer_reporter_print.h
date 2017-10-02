#pragma once

#include "caffe2/share/contrib/observers/observer_reporter.h"

namespace caffe2 {

class ObserverReporterPrint : public ObserverReporter {
 public:
  static const std::string IDENTIFIER;
  void printNet(NetBase *net, double net_delay);
  void printNetWithOperators(NetBase *net, double net_delay,
      std::vector<std::pair<std::string, double> > & operator_delays);
};
}
