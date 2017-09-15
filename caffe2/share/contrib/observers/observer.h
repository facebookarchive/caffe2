#pragma once

#include "caffe2/core/net.h"
#include "caffe2/core/timer.h"

namespace caffe2 {

class TimeNetObserver;

class TimeOperatorObserver : public ObserverBase<OperatorBase>  {
public:

  // We don't store pointer to detailedStat and instead use operator position
  // to find it through netObserver_->detailedOpStats_. Saving an extra pointer
  // this way.
  TimeOperatorObserver(OperatorBase* op, TimeNetObserver* netObserver);
  virtual ~TimeOperatorObserver();

  double getMilliseconds() const;

private:
  bool Start() override;
  bool Stop() override;

private:
  // Observer of a net that owns corresponding op. We make sure net is never
  // destructed while operator observer is still alive. First operator observer
  // gets destructed, then the op, then the net and its observer.
  // We do this trick in order to get access to net's name and other fields
  // without storing inside the operator observer. Each field is memory
  // costly here and a raw pointer is a cheapest sholution
  TimeNetObserver* netObserver_;
  double milliseconds_;
};

class TimeNetObserver : public NetObserver {
public:
  explicit TimeNetObserver(NetBase* subject_);
  virtual ~TimeNetObserver();

  caffe2::Timer& getTimer() { return timer_; }
private:
  bool Start() override;
  bool Stop() override;

  caffe2::string getObserverName(const OperatorBase *op, int idx) const;

private:
  enum LogType {
    NONE,
    OPERATOR_DELAY,
    NET_DELAY,
  };
  LogType logType_;
  unsigned int numRuns_;

  caffe2::Timer timer_;
};
}
