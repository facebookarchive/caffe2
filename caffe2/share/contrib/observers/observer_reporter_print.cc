#include "caffe2/share/contrib/observers/observer_reporter_print.h"

namespace caffe2 {

void
ObserverReporterPrint::printNet(NetBase *net, double net_delay) {
  LOG(INFO) << "Net Name: " << net->Name() << "  Net Delay: " << net_delay;
}

void
ObserverReporterPrint::printNetWithOperators(
    NetBase *net,
    double net_delay,
    std::vector<std::pair<std::string, double> > & delays) {
  printNet(net, net_delay);
  LOG(INFO) << "  Operator Delays: ";
  for (auto &p : delays) {
    LOG(INFO) << p.first << " : " << p.second;
  }
  LOG(INFO) << "===============";
}
}
