#!/usr/bin/env bash
event_name="$1"
keywords="$2"

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

echo "::group::scanning for keywords"
echo "searching for: '$keywords'"
if echo "$commit_message" | grep -qF "$keywords"; then
    result="true"
else
    result="false"
fi
echo "keywords detected: $result"
echo "::endgroup::"

echo "::set-output name=COMMIT_MESSAGE::$commit_message"
echo "::set-output name=CI_TRIGGERED::$result"
