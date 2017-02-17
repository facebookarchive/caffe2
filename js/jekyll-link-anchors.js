/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree.
 *
 * @noflow
 */
'use strict';

/* eslint comma-dangle: [1, always-multiline], prefer-object-spread/prefer-object-spread: 0 */

/* eslint-disable no-var */
/* eslint-disable prefer-arrow-callback */

// Taken and modified from https://gist.github.com/SimplGy/a229d25cdb19d7f21231
(function() {
  // Create intra-page links
  // Requires that your headings already have an `id` attribute set (because that's what jekyll
  // does). For every heading in your page, this adds a little anchor link `#` that you can click
  // to get a permalink to the heading. Ignores `h1`, because you should only have one per page.

  // This also allows us to have uniquely named links even with headings of the same name.
  // e.g.,
  //   h2: Mac  (#mac)
  //     h3: prerequisites (#mac__prerequisites)
  //   h2: Linux (#linux)
  //     h3: prerequisites (#linux__prerequisites)

  // We don't want anchors for any random h2 or h3; only ones with an
  // id attribute which aren't in h2's and h3's in the sidebar ToC and
  // header bar.
  var possibleNodeNames = ['h2', 'h3', 'h4', 'h5']; // Really try to only have up to h3, please
  var tags = document.querySelectorAll('h2[id], h3[id], h4[id], h5[id]');
  var headingNodes = Array.prototype.slice.call(tags);

  headingNodes.forEach(function(node) {
    // This is a h2 in our template h2#project_tagline
    // Just skip anchoring it if so.
    if (node.getAttribute('id') === 'project_tagline') {
      return;
    }
    var nameIdx = possibleNodeNames.indexOf(node.localName); // h2 = 0, h3 = 1, etc.
    // The anchor will be used for the actual positioning after click so that we aren't under
    // a fixed header
    var anchor;
    // The actual link of associated with the anchor
    var link;
    var id;
    var psib;
    var suffix;

    // Remove automatic id suffix added by kramdown if heading with same name exists.
    // e.g.,
    //   h2: Mac
    //     h3: prerequisites (id = prerequisites)
    //   h2: Linux
    //     h3: prerequisites (id = prerequisites-1)

    // Only match at end of string since that is where auto suffix wil be added.
    suffix = node.getAttribute('id').match(/-[0-9]+$/);
    // If the -1, etc. suffix exists, make sure someone didn't purposely put the suffix there
    // by checking against the actual text associated with the node
    if (suffix !== null &&
        node.getAttribute('id').substring(0, suffix.index) === node.textContent.toLowerCase()) {
      node.setAttribute('id', node.textContent.toLowerCase());
    }
    anchor = document.createElement('a');
    anchor.className='anchor';

    link = document.createElement('a');
    link.className = 'header-link';
    link.textContent = '#';
    id = '';

    // Avoid duplicate anchor links
    // If we are at an h3, go through the previous element siblings of this node, and find its
    // h2 parent and append it to the href text.
    psib = node.previousElementSibling;
    var idx;
    while (psib) {
      // Find the parent, if it exists.
      idx = possibleNodeNames.indexOf(psib.localName);
      if (idx !== -1 && idx === nameIdx - 1) { // if we are at h3, we want h2. That's why the - 1
        id += psib.getAttribute('id') + '__';
        break;
      }
      psib = psib.previousElementSibling;
    }
    anchor.name = id + node.getAttribute('id');
    node.insertBefore(anchor, node.firstChild);
    link.href = '#' + id + node.getAttribute('id');
    node.appendChild(link);
    // We don't want duplicate ids since the anchor will have this id already included.
    node.removeAttribute('id');
  });
})();
