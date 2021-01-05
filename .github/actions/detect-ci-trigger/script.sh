#!/usr/bin/env bash
event_name="$1"
keyword="$2"

echo "::group::fetch a sufficient number of commits"
if [[ "$event_name" == "pull_request" ]]; then
    git fetch --deepen=1 --no-tags 2>&1
else
    echo "nothing to do."
fi
echo "::endgroup::"

echo "::group::extracting the commit message"
echo "event name: $event_name"
if [[ "$event_name" == "pull_request" ]]; then
    ref="HEAD^2"
else
    ref="HEAD"
fi

commit_message="$(git log -n 1 --pretty=format:%s "$ref")"

if [[ $(echo $commit_message | wc -l) -le 1 ]]; then
    echo "commit message: '$commit_message'"
else
    echo -e "commit message:\n--- start ---\n$commit_message\n--- end ---"
fi
echo "::endgroup::"

echo "::group::scanning for the keyword"
echo "searching for: '$keyword'"
if echo "$commit_message" | grep -qF "$keyword"; then
    result="true"
else
    result="false"
fi
echo "keyword detected: $result"
echo "::endgroup::"

echo "::set-output name=COMMIT_MESSAGE::$commit_message"
echo "::set-output name=CI_TRIGGERED::$result"
