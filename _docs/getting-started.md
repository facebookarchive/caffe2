---
docid: getting-started
title: Getting Started
layout: docs-getting-started
permalink: /docs/getting-started.html
---
{% capture mac %}{% include_relative getting-started/mac.md %}{% endcapture %}
{% capture ubuntu %}{% include_relative getting-started/ubuntu.md %}{% endcapture %}
{% capture windows %}{% include_relative getting-started/windows.md %}{% endcapture %}
{% capture ios %}{% include_relative getting-started/ios.md %}{% endcapture %}
{% capture android %}{% include_relative getting-started/android.md %}{% endcapture %}

{{mac | markdownify }}
{{ubuntu | markdownify }}
{{windows | markdownify }}
{{ios | markdownify }}
{{android | markdownify }}
