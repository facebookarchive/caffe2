function operator_filter_by_query(q) {
  let operator_sections = [].slice.call(document.querySelector('.post-content').children);
  let operators = [];
  let accumulator = [];
  // Merge together all the sections into a query-able list
  for (e of operator_sections) {
    if (e.nodeName == "H2") {
      operators.push(accumulator);
      accumulator = [e]
    } else {
      accumulator.push(e);
    }
  }
  operators.push(accumulator);
  operators = operators.slice(1);

  operators.map((operator) => {
    let stringified_operator = operator.reduce((o, p) =>
		  (o.textContent || o) + (p.textContent || p)
    );
    let contains_query = stringified_operator.indexOf(q) >= 0;
		let style = "none";
    if (contains_query) {
		  style = "block";
    }
    operator.map((n) => n.style.display = style)
  });
}

window.addEventListener('load', function() {
  let query_timeout = 0;
  document.querySelector('.operator_search').addEventListener('keyup', function(e) {
    if (query_timeout) {
      clearTimeout(query_timeout);
    }
    // Don't query too fast
    query_timeout = setTimeout(function() {
      operator_filter_by_query(e.target.value);
		}, 300);
  });
});
