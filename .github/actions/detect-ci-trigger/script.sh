#!/usr/bin/env bash
event_name="$1"
keyword="$2"

echo "::group::fetch a sufficient number of commits"
echo "skipped"
# git log -n 5 2>&1
# if [[ "$event_name" == "pull_request" ]]; then
#     ref=$(git log -1 --format='%H')
#     git -c protocol.version=2 fetch --deepen=2 --no-tags --prune --progress -q origin $ref 2>&1
#     git log FETCH_HEAD
#     git checkout FETCH_HEAD
# else
#     echo "nothing to do."
# fi
# git log -n 5 2>&1
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
