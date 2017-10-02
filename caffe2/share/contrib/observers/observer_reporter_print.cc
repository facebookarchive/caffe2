#include "caffe2/share/contrib/observers/observer_reporter_print.h"

namespace caffe2 {

const std::string ObserverReporterPrint::IDENTIFIER = "Caffe2Observer : ";

void ObserverReporterPrint::printNet(NetBase* net, double net_delay) {
  LOG(INFO) << IDENTIFIER << "Net Name - " << net->Name() << " :  Net Delay - "
            << net_delay;
}

void ObserverReporterPrint::printNetWithOperators(
    NetBase* net,
    double net_delay,
    std::vector<std::pair<std::string, double>>& delays) {
  LOG(INFO) << IDENTIFIER << "Operators Delay Start";
  for (auto& p : delays) {
    LOG(INFO) << IDENTIFIER << p.first << " - " << p.second;
  }
  LOG(INFO) << IDENTIFIER << "Operators Delay End";
}
}
